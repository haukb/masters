
#Special libraries
import gurobipy as gp
from gurobipy import GRB

#Internal imports
from algorithms.mp_parallel_subproblems import MP_parallelSPs
from utils.variables_generators import make_routes_vessels_variables
from utils.misc_functions import get_same_year_nodes
from subproblems.subproblem_adhoc import Subproblem_adhoc

class MP_adhoc(MP_parallelSPs):
    def __init__(self, kwargs) -> None:
        super().__init__(kwargs)
        self._add_adhoc_extensions()
    
    def _make_subproblems(self):
        self.sp_refs = {n: Subproblem_adhoc.remote(n, self.data) for n in self.data.N}
        return

    def _add_adhoc_extensions(self):
        self._add_adhoc_variables()
        self._add_adhoc_constraints()
        self._add_adhoc_data()

        return

    def _add_adhoc_variables(self):
        T = self.data.T 
        S = self.data.S
        V = self.data.V
        R_v = self.data.R_v
        W = self.data.W

        self.variables.routes_vessels = self.m.addVars(make_routes_vessels_variables(V, R_v, T, S), vtype=GRB.CONTINUOUS, name="Routes sailed")

        return

    def _add_adhoc_constraints(self):
        #Declarations/imports for readability
        N_s = self.data.N_s
        V = self.data.V
        T = self.data.T 
        S = self.data.S
        R_v = self.data.R_v

        BETA = self.data.BETA
        TIMEPERIOD_DURATION = self.data.TIMEPERIOD_DURATION
        ROUTE_SAILING_TIME = self.data.ROUTE_SAILING_TIME

        vessels = self.variables.vessels
        routes_vessels = self.variables.routes_vessels


        self.constraints.c13 = self.m.addConstrs(gp.quicksum(BETA.iloc[n,t]*vessels[(v, n)] for n in N_s[s]) >= \
            (1/TIMEPERIOD_DURATION.iloc[0,0])*gp.quicksum(ROUTE_SAILING_TIME.iloc[v,r]*routes_vessels[(v,r,t,s)] for r in R_v[v]) \
                for v in V for t in T for s in S)
        
        return

    def _add_adhoc_data(self):
        self.data.routes_vessels = {}

        for v in self.data.V:
                for r in self.data.R_v[v]:
                    for t in self.data.T:
                        for s in self.data.S:
                            self.data.routes_vessels[(v,r,t,s)] = []

        return

    def _add_cut(self, N, sp_data):
        m = self.m

        #Declarations/imports for readability
        V = self.data.V
        P = self.data.P
        NP_n = self.data.NP_n
        S_n = self.data.S_n
        T_n = self.data.T_n
        R_v = self.data.R_v

        for n in N: 
            lhs = self.variables.phi[n]
            z_sub = sp_data[n].obj_vals[-1]
            sens_vessels = sp_data[n].sens_vessels[-1]
            sens_ports = sp_data[n].sens_ports[-1]
            sens_routes_vessels = sp_data[n].sens_routes_vessels[-1]
            rhs = (z_sub + 
            gp.quicksum(sens_vessels[v] * gp.quicksum(self.variables.vessels[(v,m)]-self.data.vessels[(v,m)][-1] for m in NP_n[n]) for v in V) + 
            gp.quicksum(sens_ports[i] * gp.quicksum(self.variables.ports[(i,m)]-self.data.ports[(i,m)][-1] for m in NP_n[n]) for i in P[1:]) +
            gp.quicksum(sens_routes_vessels[v][r,t,s]*(self.variables.routes_vessels[(v,r,t,s)]-self.data.routes_vessels[(v,r,t,s)][-1]) for v in V for r in R_v[v] for t in T_n[n] for s in S_n[n]))
            m.addConstr(lhs >= rhs)

        return
    
    def _save_vars(self, model=None):
        #First save the routing variables spesific to the adhoc implementation
        #Fixing the routing variables
        if model == None:
            model = self

        for v in self.data.V:
            for r in self.data.R_v[v]:
                for t in self.data.T:
                    for s in self.data.S:
                        if model == None:
                            s_solved = s
                        else: 
                            s_solved = model.data.S[0]
                        
                        self.data.routes_vessels[(v,r,t,s)].append(model.variables.routes_vessels[(v,r,t,s_solved)].x)
        
    
        #Then save the other variables
        super()._save_vars(model)

        return 

    