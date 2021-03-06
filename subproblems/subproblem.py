# Import Gurobi Library
import gurobipy as gp
from gurobipy import GRB

from itertools import product

# Import helper functions
from utils.variables_generators import (
    make_delivery_vessel_variables,
    make_delivery_truck_variables,
    make_routes_vessels_variables,
)
from utils.misc_functions import get_same_year_nodes


# Class which can have attributes set.
class expando(object):
    pass


# Subproblem
class Subproblem:
    def __init__(self, NODE, mp=None, data=None, PARALLEL=False):
        self.NODE = NODE
        if PARALLEL:
            self.mp = expando()
            self.mp.data = data
        else:
            self.mp = mp
        self.PARALLEL = PARALLEL
        self.data = expando()
        self.variables = expando()
        self.constraints = expando()
        self.results = expando()
        self._load_data()
        self._build_model()

    def solve(self):
        self.m.optimize()
        self._save_vars()
        if self.PARALLEL is True:
            return self.data
        else:
            return

    def _load_data(self):
        # make the selection of scenarios and years for the given subproblem
        self.data.S_star = self.mp.data.S_n[self.NODE]
        self.data.T_star = self.mp.data.T_n[self.NODE]

        self.data.sens_vessels = []
        self.data.sens_ports = []
        self.data.sens_routes_vessels = []
        self.data.obj_vals = []
        self.data.delivery_vessel = []
        self.data.delivery_truck = []
        self.data.pickup_vessel = []
        self.data.pickup_truck = []
        self.data.routes_vessels = []

        return

    ###
    #   Model Building
    ###
    def _build_model(self):
        self.m = gp.Model(
            env=gp.Env(
                params={
                    "OutputFlag": 0,
                }
            )
        )
        self._build_variables()
        self._build_objective()
        self._build_constraints()
        self.m.update()

        return

    def _build_variables(self):
        m = self.m
        # fetch the selection of scenarios and years for the given subproblem
        S = self.data.S_star
        T = self.data.T_star

        # Fetch the general sets from the mp
        V = self.mp.data.V
        P = self.mp.data.P
        P_r = self.mp.data.P_r
        N = self.mp.data.N
        R_v = self.mp.data.R_v
        K_i = self.mp.data.K_i
        W = self.mp.data.W

        # self.variables.routes_vessels = m.addVars(make_routes_vessels_variables(V, R_v, T, S), vtype=GRB.CONTINUOUS, name="Routes sailed")
        self.variables.routes_vessels = m.addVars(
            make_routes_vessels_variables(V, R_v, self.mp.data.T, self.mp.data.S),
            vtype=GRB.CONTINUOUS,
            name="Routes sailed",
        )
        self.variables.delivery_vessel = m.addVars(
            make_delivery_vessel_variables(V, R_v, P_r, T, W, S),
            vtype=GRB.CONTINUOUS,
            name="Delivery by vessel",
        )
        self.variables.pickup_vessel = m.addVars(
            make_delivery_vessel_variables(V, R_v, P_r, T, W, S),
            vtype=GRB.CONTINUOUS,
            name="Pick-up by vessels",
        )
        self.variables.delivery_truck = m.addVars(
            make_delivery_truck_variables(P, K_i, T, W, S),
            vtype=GRB.CONTINUOUS,
            name="Delivery by truck",
        )
        self.variables.pickup_truck = m.addVars(
            make_delivery_truck_variables(P, K_i, T, W, S),
            vtype=GRB.CONTINUOUS,
            name="Pick-up by truck",
        )
        self.variables.load = m.addVars(
            make_delivery_vessel_variables(V, R_v, P_r, T, W, S),
            vtype=GRB.CONTINUOUS,
            name="Vessel load",
        )

        # Vessel and port variables which will be fixed by master problem
        self.variables.vessels_free = m.addVars(
            product(V, N), vtype=GRB.CONTINUOUS, name="Vessel investment"
        )
        self.variables.ports_free = m.addVars(
            product(P, N), vtype=GRB.CONTINUOUS, name="Port investment"
        )

        m.update()

        return

    def _build_objective(self):
        m = self.m

        # Fetch the selection of scenarios and years for the given subproblem
        S = self.data.S_star
        T = self.data.T_star

        # Fetch sets from mp
        V = self.mp.data.V
        P_r = self.mp.data.P_r
        P_k = self.mp.data.P_k
        R_v = self.mp.data.R_v
        K = self.mp.data.K
        W = self.mp.data.W

        # Fetch data from mp
        PROB_SCENARIO = self.mp.data.PROB_SCENARIO
        SAILING_COST = self.mp.data.SAILING_COST
        PORT_HANDLING = self.mp.data.PORT_HANDLING
        TRUCK_COST = self.mp.data.TRUCK_COST
        NUM_WEEKS = self.mp.data.NUM_WEEKS

        # Fetch variables
        delivery_truck = self.variables.delivery_truck
        delivery_vessel = self.variables.delivery_vessel
        pickup_truck = self.variables.pickup_truck
        pickup_vessel = self.variables.pickup_vessel
        routes_vessels = self.variables.routes_vessels

        # Make subproblem objective function expression
        vessel_opex = (52 / NUM_WEEKS) * gp.quicksum(
            PROB_SCENARIO.iloc[0, s]
            * gp.quicksum(
                SAILING_COST[v, r, t] * routes_vessels[(v, r, t, s)]
                + gp.quicksum(
                    PORT_HANDLING[i, t]
                    * (
                        delivery_vessel[(i, v, r, t, w, s)]
                        + pickup_vessel[(i, v, r, t, w, s)]
                    )
                    for i in P_r[r]
                )
                for v in V
                for r in R_v[v]
            )
            for t in T
            for w in W
            for s in S
        )

        truck_opex = (52 / NUM_WEEKS) * gp.quicksum(
            PROB_SCENARIO.iloc[0, s]
            * TRUCK_COST[i, k, t, s]
            * (delivery_truck[(i, k, t, w, s)] + pickup_truck[(i, k, t, w, s)])
            for k in K
            for i in P_k[k]
            for t in T
            for w in W
            for s in S
        )

        m.setObjective(vessel_opex + truck_opex, gp.GRB.MINIMIZE)

    def _build_constraints(self):
        # Fetch model
        m = self.m

        # Fetch the selection of scenarios and years for the given subproblem
        S = self.data.S_star
        T = self.data.T_star

        # Fetch the general sets from the mp
        V = self.mp.data.V
        P = self.mp.data.P
        P_r = self.mp.data.P_r
        P_k = self.mp.data.P_k
        N = self.mp.data.N
        N_s = self.mp.data.N_s
        R_v = self.mp.data.R_v
        R_vi = self.mp.data.R_vi
        K = self.mp.data.K
        K_i = self.mp.data.K_i
        W = self.mp.data.W
        A_r = self.mp.data.A_r

        # Fetch data
        DELIVERY = self.mp.data.DELIVERY
        PICKUP = self.mp.data.PICKUP
        BETA = self.mp.data.BETA
        PORTS = self.mp.data.PORTS
        VESSELS = self.mp.data.VESSELS
        ROUTE_SAILING_TIME = self.mp.data.ROUTE_SAILING_TIME
        TIMEPERIOD_DURATION = self.mp.data.TIMEPERIOD_DURATION

        # Fetch variables
        delivery_truck = self.variables.delivery_truck
        delivery_vessel = self.variables.delivery_vessel
        pickup_truck = self.variables.pickup_truck
        pickup_vessel = self.variables.pickup_vessel
        vessels_free = self.variables.vessels_free
        routes_vessels = self.variables.routes_vessels
        ports_free = self.variables.ports_free
        load = self.variables.load

        self.constraints.cuts = {}

        self.constraints.c1 = m.addConstrs(
            gp.quicksum(delivery_truck[(i, k, t, w, s)] for i in P_k[k])
            >= DELIVERY[k, t, w, s]
            for k in K
            for t in T
            for w in W
            for s in S
        )

        self.constraints.c2 = m.addConstrs(
            gp.quicksum(pickup_truck[(i, k, t, w, s)] for i in P_k[k])
            >= PICKUP[k, t, w, s]
            for k in K
            for t in T
            for w in W
            for s in S
        )

        self.constraints.c3 = m.addConstrs(
            gp.quicksum(
                delivery_vessel[(i, v, r, t, w, s)] for v in V for r in R_vi[v, i]
            )
            <= gp.quicksum(DELIVERY[k, t, w, s] for k in K_i[i])
            * gp.quicksum(BETA.iloc[n, t] * ports_free[(i, n)] for n in N_s[s])
            for i in P[1:]
            for t in T
            for w in W
            for s in S
        )

        self.constraints.c4 = m.addConstrs(
            gp.quicksum(
                pickup_vessel[(i, v, r, t, w, s)] for v in V for r in R_vi[v, i]
            )
            <= gp.quicksum(PICKUP[k, t, w, s] for k in K_i[i])
            * gp.quicksum(BETA.iloc[n, t] * ports_free[(i, n)] for n in N_s[s])
            for i in P[1:]
            for t in T
            for w in W
            for s in S
        )

        self.constraints.c5 = m.addConstrs(
            pickup_vessel[(0, v, r, t, w, s)]
            - gp.quicksum(delivery_vessel[(i, v, r, t, w, s)] for i in P_r[r][1:])
            == 0
            for v in V
            for r in R_v[v]
            for t in T
            for w in W
            for s in S
        )

        self.constraints.c6 = m.addConstrs(
            delivery_vessel[(0, v, r, t, w, s)]
            - gp.quicksum(pickup_vessel[(i, v, r, t, w, s)] for i in P_r[r][1:])
            == 0
            for v in V
            for r in R_v[v]
            for t in T
            for w in W
            for s in S
        )

        self.constraints.c7 = m.addConstrs(
            gp.quicksum(
                delivery_vessel[(i, v, r, t, w, s)] for v in V for r in R_vi[v, i]
            )
            - gp.quicksum(delivery_truck[(i, k, t, w, s)] for k in K_i[i])
            == 0
            for i in P[1:]
            for t in T
            for w in W
            for s in S
        )

        self.constraints.c8 = m.addConstrs(
            gp.quicksum(
                pickup_vessel[(i, v, r, t, w, s)] for v in V for r in R_vi[v, i]
            )
            - gp.quicksum(pickup_truck[(i, k, t, w, s)] for k in K_i[i])
            == 0
            for i in P[1:]
            for t in T
            for w in W
            for s in S
        )

        self.constraints.c9 = m.addConstrs(
            load[(j, v, r, t, w, s)]
            - (
                load[(i, v, r, t, w, s)]
                + pickup_vessel[(j, v, r, t, w, s)]
                - delivery_vessel[(j, v, r, t, w, s)]
            )
            == 0
            for v in V
            for r in R_v[v]
            for (i, j) in A_r[r][:-1]
            for t in T
            for w in W
            for s in S
        )

        self.constraints.c10 = m.addConstrs(
            load[(j, v, r, t, w, s)]
            <= VESSELS.iloc[v, 0] * routes_vessels[(v, r, t, s)]
            for v in V
            for r in R_v[v]
            for j in P_r[r]
            for t in T
            for w in W
            for s in S
        )

        self.constraints.c11 = m.addConstrs(
            pickup_vessel[(0, v, r, t, w, s)]
            <= VESSELS.iloc[v, 0] * routes_vessels[(v, r, t, s)]
            for v in V
            for r in R_v[v]
            for t in T
            for w in W
            for s in S
        )
        # Ensuring that the different ports are visited with respects to their frequency demand and totalt amount of time needed to sail routes does not exceed the time available with the current fleet
        """REMOVED FROM SUBPROBLEMS - to avoid infeasible solutions
        self.constraints.c12 = m.addConstrs(gp.quicksum(routes_vessels[(v,r,t,s)] for v in V for r in R_vi[v,i]) >= 
        PORTS.iloc[i,2]*gp.quicksum(BETA.iloc[n,t]*ports_free[(i, n)] for n in N_s[s]) 
        for i in P for t in T for s  in S)
        """

        self.constraints.c13 = m.addConstrs(
            gp.quicksum(BETA.iloc[n, t] * vessels_free[(v, n)] for n in N_s[s])
            >= (1 / TIMEPERIOD_DURATION.iloc[0, 0])
            * gp.quicksum(
                ROUTE_SAILING_TIME.iloc[v, r] * routes_vessels[(v, r, t, s)]
                for r in R_v[v]
            )
            for v in V
            for t in T
            for s in S
        )

        self.constraints.c14 = m.addConstrs(
            load[(0, v, r, t, w, s)] - pickup_vessel[(0, v, r, t, w, s)] == 0
            for v in V
            for r in R_v[v]
            for t in T
            for w in W
            for s in S
        )

        self.constraints.fix_vessels = m.addConstrs(
            vessels_free[(v, n)] == 0 for v in V for n in N
        )

        self.constraints.fix_ports = m.addConstrs(
            ports_free[(i, n)] == 0 for i in P for n in N
        )

        return

    def _save_vars(self):
        V = self.mp.data.V
        P = self.mp.data.P
        # Save the objective value
        self.data.obj_vals.append(self.m.ObjVal)
        # Save the sensitivity values
        self.data.sens_vessels.append(
            [self.constraints.fix_vessels[(v, self.NODE)].pi for v in V]
        )
        self.data.sens_ports.append(
            [self.constraints.fix_ports[(i, self.NODE)].pi for i in P]
        )
        """ CURRENTLY IT MAKES THE MEMORY EXPLODE TO SAVE ALL THESE VARIABLES
        # Save the variable values
        routes_vessels = {}
        for k, v in self.variables.routes_vessels.iteritems():
            routes_vessels[k] = v.x
        self.data.routes_vessels.append(routes_vessels)

        delivery_vessel = {}
        for k, v in self.variables.delivery_vessel.iteritems():
            delivery_vessel[k] = v.x
        self.data.delivery_vessel.append(delivery_vessel)

        pickup_vessel = {}
        for k, v in self.variables.pickup_vessel.iteritems():
            pickup_vessel[k] = v.x
        self.data.pickup_vessel.append(pickup_vessel)

        delivery_truck = {}
        for k, v in self.variables.delivery_truck.iteritems():
            delivery_truck[k] = v.x
        self.data.delivery_truck.append(delivery_truck)

        pickup_truck = {}
        for k, v in self.variables.pickup_truck.iteritems():
            pickup_truck[k] = v.x
        self.data.pickup_truck.append(pickup_truck)
        """

        return

    def _update_fixed_vars(self):

        for n in self.mp.data.N:
            for i in self.mp.data.P:
                port_val = self.mp.data.ports[(i, n)][-1]
                self.constraints.fix_ports[(i, n)].rhs = port_val
            for v in self.mp.data.V:
                vessel_val = self.mp.data.vessels[(v, n)][-1]
                self.constraints.fix_vessels[(v, n)].rhs = vessel_val

        return

    def _update_fixed_vars_callback(self, model=None):

        if model is None:
            for n in self.mp.data.N:
                for i in self.mp.data.P:
                    port_val = self.mp.m.cbGetSolution(self.mp.variables.ports[(i, n)])
                    self.constraints.fix_ports[(i, n)].rhs = port_val
                for v in self.mp.data.V:
                    vessel_val = self.mp.m.cbGetSolution(
                        self.mp.variables.vessels[(v, n)]
                    )
                    self.constraints.fix_vessels[(v, n)].rhs = vessel_val

        else:  # Only used for the hot start iteration
            for n_solved in self.mp.data.N_s[model.data.S[0]]:
                N = get_same_year_nodes(
                    n_solved, self.mp.data.N, self.mp.data.YEAR_OF_NODE
                )
                for n in N:
                    for i in self.mp.data.P:
                        self.constraints.fix_ports[(i, n)].rhs = model.variables.ports[
                            (i, n_solved)
                        ].x
                    for v in self.mp.data.V:
                        self.constraints.fix_vessels[
                            (v, n)
                        ].rhs = model.variables.vessels[(v, n_solved)].x

        return
