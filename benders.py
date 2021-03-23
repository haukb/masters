# Import Gurobi Library
import gurobipy as gb


# Class which can have attributes set.
class expando(object):
    pass


# Master problem
class Benders_Master:
    def __init__(self, benders_gap=0.001, max_iters=10):
        self.max_iters = max_iters
        self.data = expando()
        self.variables = expando()
        self.constraints = expando()
        self.results = expando()
        self.count_loops = 0
        self._load_data(benders_gap=benders_gap)
        self._build_model()

    def optimize(self, simple_results=False):
        # Initial solution
        self.model.optimize()
        # Build subproblem from solution
        self.submodel = Benders_Subproblem(self)
        self.submodel.update_fixed_vars(self)
        self.submodel.optimize()
        self._add_cut()
        self._update_bounds()
        self._save_vars()
        #count_loops = 1
        print(f'Upper bound {self.data.ub} \n Lower bound {self.data.lb}')
        while self.data.ub > self.data.lb + self.data.benders_gap and len(self.data.cutlist) < self.max_iters:
            self.model.optimize()
            self.submodel.update_fixed_vars(self)
            self.submodel.optimize()
            self._add_cut()
            self._update_bounds()
            self._save_vars()
            self.count_loops += 1
        #print(f'Number of loops: {count_loops}')
        pass

    ###
    #   Loading functions
    ###

    def _load_data(self, benders_gap=0.001):
        self.data.cutlist = []
        self.data.upper_bounds = []
        self.data.lower_bounds = []
        self.data.lambdas = {}
        self.data.benders_gap = benders_gap
        self.data.ub = gb.GRB.INFINITY
        self.data.lb = -gb.GRB.INFINITY
        self.data.zs = []
        self.data.ys = []
        self.data.u1s = []
        self.data.u2s = []

    ###
    #   Model Building
    ###
    def _build_model(self):
        self.model = gb.Model()
        self._build_variables()
        self._build_objective()
        self._build_constraints()
        self.model.update()

    def _build_variables(self):
        m = self.model

        self.variables.z = m.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY, vtype = gb.GRB.CONTINUOUS, name='x')
        self.variables.y = m.addVar(lb=0.0, ub=gb.GRB.INFINITY, vtype = gb.GRB.INTEGER, name='y')
        m.update()

    def _build_objective(self):
        self.model.setObjective(self.variables.z, gb.GRB.MINIMIZE)

    def _build_constraints(self):
        self.constraints.cuts = {}
        pass

    ###
    # Cut adding
    ###
    def _add_cut(self):
        z = self.variables.z
        y = self.variables.y
        cut = len(self.data.cutlist)
        self.data.cutlist.append(cut)
        # Get sensitivity from subproblem
        u1 = self.submodel.variables.u1.x
        u2 = self.submodel.variables.u2.x

        if self.count_loops == 2:
            y = 100

        self.constraints.cuts[cut] = self.model.addConstr(z >= 4*y + (7*u1 + 6*u2) - (1*u1 + 2*u2)*y)

    ###
    # Update upper and lower bounds
    ###
    def _update_bounds(self):
        z_sub = self.submodel.model.ObjVal
        z_master = self.model.ObjVal
        #print(f'z subproblem {z_sub} \n z masterproblem {z_master}')
        self.data.ub = z_master + z_sub
        if self.data.ub<0:
            self.data.ub = 1000
        # The best lower bound is the current bestbound,
        # This will equal z_master at optimality
        self.data.lb = z_master
        self.data.upper_bounds.append(self.data.ub)
        self.data.lower_bounds.append(self.data.lb)

    def _save_vars(self):
        self.data.zs.append(self.variables.z.x)
        self.data.ys.append(self.variables.y.x)
        self.data.u1s.append(self.submodel.variables.u1.x)
        self.data.u2s.append(self.submodel.variables.u2.x)


# Subproblem
class Benders_Subproblem:
    def __init__(self, MP):
        self.data = expando()
        self.data.MP = MP
        self.variables = expando()
        self.constraints = expando()
        self.results = expando()
        self._build_model()
        self.update_fixed_vars()

    def optimize(self):
        self.model.optimize()

    ###
    #   Model Building
    ###
    def _build_model(self):
        self.model = gb.Model()
        self._build_variables()
        self._build_objective()
        self._build_constraints()
        self.model.update()

    def _build_variables(self):
        m = self.model

        # Power flow on line l
        self.variables.u1 = m.addVar(lb=0.0, ub=gb.GRB.INFINITY,vtype = gb.GRB.CONTINUOUS, name='u1')
        self.variables.u2 = m.addVar(lb=0, ub=gb.GRB.INFINITY, vtype = gb.GRB.CONTINUOUS, name='u2')
        self.variables.y_free = m.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY, vtype = gb.GRB.INTEGER, name='y_free')

        m.update()

    def _build_objective(self):
        m = self.model

        self.model.setObjective(
            (7*self.variables.u1 + 6*self.variables.u2) - (1*self.variables.u1 + 2*self.variables.u2)
            *self.variables.y_free,
            gb.GRB.MAXIMIZE)

    def _build_constraints(self):
        m = self.model
        u1 = self.variables.u1
        u2 = self.variables.u2

        self.constraints.c1 = m.addConstr(3*u1 + 1*u2 <= 15)
        self.constraints.c2 = m.addConstr(1*u1 + 5*u2 <= 10)
        self.constraints.fix_y = m.addConstr(self.variables.y_free == 0)

    def update_fixed_vars(self, MP=None):
        if MP is None:
            MP = self.data.MP
        self.constraints.fix_y.rhs = MP.variables.y.x

if __name__ == '__main__':
    print('Start')
    m = Benders_Master()
    m.optimize()
    print('Stop')