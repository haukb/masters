# Standard libraries

# Special libraries
import ray

# Internal imports
from subproblems.subproblem import Subproblem


# Subproblem
@ray.remote
class Subproblem_adhoc(Subproblem):
    def __init__(self, NODE, data) -> None:
        super().__init__(NODE=NODE, data=data, PARALLEL=True)
        self._add_adhoc_extensions()

    def _add_adhoc_extensions(self):
        self._add_route_fixation_constraint()
        return

    def _save_vars(self):
        V = self.mp.data.V
        R_v = self.mp.data.R_v
        T = self.data.T_star
        S = self.data.S_star

        sens_routes_vessels = {}

        for v in V:
            for r in R_v[v]:
                for t in T:
                    for s in S:
                        sens_routes_vessels[
                            (v, r, t, s)
                        ] = self.constraints.fix_routes_vessels[(v, r, t, s)].pi
        self.data.sens_routes_vessels.append(sens_routes_vessels)

        super()._save_vars()

        return

    def _add_route_fixation_constraint(self):

        T = self.data.T_star
        S = self.data.S_star
        V = self.mp.data.V
        R_v = self.mp.data.R_v

        self.constraints.fix_routes_vessels = self.m.addConstrs(
            self.variables.routes_vessels[(v, r, t, s)] == 0
            for v in V
            for r in R_v[v]
            for t in T
            for s in S
        )

        return

    def _update_fixed_vars(self, updated_data):
        T = self.data.T_star
        S = self.data.S_star
        N = self.mp.data.N
        P = self.mp.data.P
        V = self.mp.data.V
        R_v = self.mp.data.R_v

        # Fixing the port and vessel variables
        for n in N:
            for i in P:
                port_val = updated_data.ports[(i, n)][-1]
                self.constraints.fix_ports[(i, n)].rhs = port_val
            for v in V:
                vessel_val = updated_data.vessels[(v, n)][-1]
                self.constraints.fix_vessels[(v, n)].rhs = vessel_val

        # Fixing the routing variables
        for v in V:
            for r in R_v[v]:
                for t in T:
                    for s in S:
                        routes_vessels_val = updated_data.routes_vessels[-1][
                            (v, r, t, s)
                        ]
                        self.constraints.fix_routes_vessels[
                            (v, r, t, s)
                        ].rhs = routes_vessels_val

        return
