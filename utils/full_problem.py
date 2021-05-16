# Import Gurobi Library
import gurobipy as gp
from gurobipy import GRB
from time import time

from itertools import product

# Import helper functions
from .variables_generators import (
    make_delivery_vessel_variables,
    make_delivery_truck_variables,
    make_routes_vessels_variables,
    make_weekly_routes_vessels_variables,
)


# Class which can have attributes set.
class expando(object):
    pass


# Subproblem for the full problem with expected values instead of stochastic variables
class Full_problem:
    def __init__(self, mp, SCENARIOS=None):
        self.mp = mp
        self.data = expando()
        self.SCENARIOS = SCENARIOS

        if SCENARIOS is None:
            self.data.S = self.mp.data.S
        else:
            self.data.S = SCENARIOS

        self.variables = expando()
        self.constraints = expando()
        self.results = expando()
        self._build_model()

    def solve(self):
        self.m.setParam("TimeLimit", self.mp.data.TIME_LIMIT)
        t0 = time()
        self.m.optimize()
        t1 = time()
        self.data.solve_time = t1 - t0
        return

    ###
    #   Model Building
    ###
    def _build_model(self):
        self.m = gp.Model()
        # self.m.setParam('TimeLimit', 120)
        self.m.setParam("OutputFlag", 0)  # Suppress default output
        self._build_variables()
        self._build_objective()
        self._build_constraints()
        self.m.update()

        return

    def _build_variables(self):
        m = self.m
        # Fetch the scenario(s) for which to solve the full problem
        S = self.data.S

        # Fetch the general sets from the mp
        V = self.mp.data.V
        P = self.mp.data.P
        P_r = self.mp.data.P_r
        if self.SCENARIOS is None:  # Solving for all scenarios
            N = self.mp.data.N
        else:
            N = self.mp.data.N_s[S[0]]  # Solving deterministic 1-scenario problem
        R_v = self.mp.data.R_v
        K_i = self.mp.data.K_i
        W = self.mp.data.W
        T = self.mp.data.T

        self.variables.vessels = m.addVars(
            product(V, N), vtype=GRB.INTEGER, name="Vessel investment"
        )
        self.variables.ports = m.addVars(
            product(P, N), vtype=GRB.BINARY, name="Port investment"
        )
        if self.mp.data.WEEKLY_ROUTING == True:
            self.variables.routes_vessels = m.addVars(
                make_weekly_routes_vessels_variables(V, R_v, T, W, S),
                vtype=GRB.CONTINUOUS,
                name="Routes sailed",
            )
        else:
            self.variables.routes_vessels = m.addVars(
                make_routes_vessels_variables(V, R_v, T, S),
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

        m.update()

    def _build_constraints(self):
        # Fetch model
        m = self.m

        # Fetch the selection of scenarios and years for the given subproblem
        S = self.data.S

        # Fetch the general sets from the mp
        V = self.mp.data.V
        P = self.mp.data.P
        P_r = self.mp.data.P_r
        P_k = self.mp.data.P_k
        N_s = self.mp.data.N_s
        R_v = self.mp.data.R_v
        R_vi = self.mp.data.R_vi
        K = self.mp.data.K
        K_i = self.mp.data.K_i
        W = self.mp.data.W
        T = self.mp.data.T
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
        vessels = self.variables.vessels
        ports = self.variables.ports
        delivery_truck = self.variables.delivery_truck
        delivery_vessel = self.variables.delivery_vessel
        pickup_truck = self.variables.pickup_truck
        pickup_vessel = self.variables.pickup_vessel
        routes_vessels = self.variables.routes_vessels
        load = self.variables.load

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
            * gp.quicksum(BETA.iloc[n, t] * ports[(i, n)] for n in N_s[s])
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
            * gp.quicksum(BETA.iloc[n, t] * ports[(i, n)] for n in N_s[s])
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
        # self.constraints.c12 = m.addConstrs(gp.quicksum(routes_vessels[(v,r,t,s)] for v in V for r in R_vi[v,i]) >=
        # PORTS.iloc[i,2]*gp.quicksum(BETA.iloc[n,t]*ports[(i, n)] for n in N_s[s])
        # for i in P for t in T for s in S)

        self.constraints.c13 = m.addConstrs(
            gp.quicksum(BETA.iloc[n, t] * vessels[(v, n)] for n in N_s[s])
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

        return

    def _build_objective(self):
        m = self.m

        # Fetch the selection of scenarios and years for the given subproblem
        S = self.data.S

        # Fetch sets from mp
        V = self.mp.data.V
        P = self.mp.data.P
        N_s = self.mp.data.N_s
        P_r = self.mp.data.P_r
        P_k = self.mp.data.P_k
        R_v = self.mp.data.R_v
        K = self.mp.data.K
        W = self.mp.data.W
        T = self.mp.data.T

        # Fetch data from mp
        PROB_SCENARIO = self.mp.data.PROB_SCENARIO
        VESSEL_INVESTMENT = self.mp.data.VESSEL_INVESTMENT
        PORT_INVESTMENT = self.mp.data.PORT_INVESTMENT
        SAILING_COST = self.mp.data.SAILING_COST
        PORT_HANDLING = self.mp.data.PORT_HANDLING
        TRUCK_COST = self.mp.data.TRUCK_COST
        NUM_WEEKS = self.mp.data.NUM_WEEKS

        # Fetch variables
        vessels = self.variables.vessels
        ports = self.variables.ports
        delivery_truck = self.variables.delivery_truck
        delivery_vessel = self.variables.delivery_vessel
        pickup_truck = self.variables.pickup_truck
        pickup_vessel = self.variables.pickup_vessel
        routes_vessels = self.variables.routes_vessels

        # Calculate the expression for both investment and opertational cost
        self.data.investment_costs = investment_costs = gp.quicksum(
            PROB_SCENARIO.iloc[0, s]
            * (
                gp.quicksum(
                    gp.quicksum(VESSEL_INVESTMENT[v, n] * vessels[(v, n)] for v in V)
                    + gp.quicksum(PORT_INVESTMENT[i, n] * ports[(i, n)] for i in P)
                    for n in N_s[s]
                )
            )
            for s in S
        )

        self.data.vessel_opex = vessel_opex = gp.quicksum(
            PROB_SCENARIO.iloc[0, s]
            * (52 / NUM_WEEKS)
            * gp.quicksum(
                gp.quicksum(
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
            )
            for s in S
        )

        self.data.truck_opex = truck_opex = gp.quicksum(
            PROB_SCENARIO.iloc[0, s]
            * (52 / NUM_WEEKS)
            * gp.quicksum(
                TRUCK_COST[i, k, t, s]
                * (delivery_truck[(i, k, t, w, s)] + pickup_truck[(i, k, t, w, s)])
                for k in K
                for i in P_k[k]
                for t in T
                for w in W
            )
            for s in S
        )

        self.data.total_costs = total_costs = (
            investment_costs + truck_opex + vessel_opex
        )

        m.setObjective(total_costs, gp.GRB.MINIMIZE)

        return
