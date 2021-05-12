#Standard libraries
from itertools import product

#Special libraries
import ray

#Internal imports
from subproblems.subproblem import Subproblem

# Subproblem
@ray.remote
class Subproblem_adhoc(Subproblem):
    def __init__(self, NODE, data) -> None:
        super().__init__(NODE = NODE, data=data, PARALLEL=True)
        self._add_adhoc_extensions()

    def _add_adhoc_extensions(self):
        self._add_route_fixation_constraint()
        return
    
    def _save_vars(self):
        V = self.mp.data.V
        R_v = self.mp.data.R_v
        T_n = self.mp.data.T_n
        W = self.mp.data.W
        S_n = self.mp.data.S_n

        #sens_routes_vessels = [dict.fromkeys(product(R_v[v],T_n[self.NODE], S_n[self.NODE]), []) for v in V]
        sens_routes_vessels = [dict.fromkeys(product(R_v[v],self.mp.data.T, self.mp.data.S), []) for v in V]
        
        #Fixing the routing variables
        for v in V:
            for r in R_v[v]:
                for t in self.mp.data.T:#T_n[self.NODE]:
                    for s in self.mp.data.S:#S_n[self.NODE]:
                        sens_routes_vessels[v][(r,t,s)] = self.constraints.fix_routes_vessels[(v,r,t,s)].pi

        self.data.sens_routes_vessels.append(sens_routes_vessels)

        super()._save_vars()

        return

    def _add_route_fixation_constraint(self):

        T_n = self.mp.data.T_n
        S_n = self.mp.data.S_n
        V = self.mp.data.V
        R_v = self.mp.data.R_v
        W = self.mp.data.W

        routes_vessels_free = self.variables.routes_vessels

        #self.constraints.fix_routes_vessels = self.m.addConstrs(routes_vessels_free[(v,r,t,s)] == 0 for v in V for r in R_v[v] for t in T_n[self.NODE] for s in S_n[self.NODE])
        self.constraints.fix_routes_vessels = self.m.addConstrs(routes_vessels_free[(v,r,t,s)] == 0 for v in V for r in R_v[v] for t in self.mp.data.T for s in self.mp.data.S)

        return

    def update_fixed_vars(self, updated_data):
        
        #Fixing the port and vessel variables
        for n in self.mp.data.N:
            for i in self.mp.data.P:
                port_val = updated_data.ports[(i,n)][-1]
                self.constraints.fix_ports[(i,n)].rhs = port_val
            for v in self.mp.data.V:
                vessel_val = updated_data.vessels[(v,n)][-1]
                self.constraints.fix_vessels[(v,n)].rhs = vessel_val
                    
        #Fixing the routing variables
        for v in self.mp.data.V:
            for r in self.mp.data.R_v[v]:
                for t in self.mp.data.T:#_n[self.NODE]:
                    for s in self.mp.data.S:#_n[self.NODE]:
                        routes_vessels_val = updated_data.routes_vessels[(v,r,t,s)][-1]
                        self.constraints.fix_routes_vessels[(v,r,t,s)].rhs = routes_vessels_val

        return

    

    