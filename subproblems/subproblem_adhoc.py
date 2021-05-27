# Standard libraries

# Special libraries
import gurobipy as gp
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
        PORT_HANDLING = self.mp.data.PORT_HANDLING
        TRUCK_COST = self.mp.data.TRUCK_COST
        NUM_WEEKS = self.mp.data.NUM_WEEKS

        # Fetch variables
        delivery_truck = self.variables.delivery_truck
        delivery_vessel = self.variables.delivery_vessel
        pickup_truck = self.variables.pickup_truck
        pickup_vessel = self.variables.pickup_vessel

        # Make subproblem objective function expression
        vessel_opex = (52 / NUM_WEEKS) * gp.quicksum(
            PROB_SCENARIO.iloc[0, s]
            * gp.quicksum(
                PORT_HANDLING[i, t]
                * (
                    delivery_vessel[(i, v, r, t, w, s)]
                    + pickup_vessel[(i, v, r, t, w, s)]
                )
                for v in V
                for r in R_v[v]
                for i in P_r[r]
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
