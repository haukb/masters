# Import libraries
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
from itertools import product
from time import time

# Import other classes
from utils.full_problem import Full_problem
from subproblems.subproblem import Subproblem

# Import helper functions
from utils.special_set_generators import (
    beta_set_generator,
    arc_set_generator,
    port_route_set_generator,
    route_vessel_set_generator,
    route_vessel_port_set_generator,
    port_customer_set_generator,
    scenario_node_set_generator,
    year_node_set_generator,
    parent_node_set_generator,
)
from utils.cost_generators import (
    vessel_investment,
    port_investment,
    sailing_cost,
    truck_cost,
    port_handling_cost,
)
from utils.short_term_uncertainty import draw_weekly_demand
from utils.misc_functions import get_same_year_nodes
from utils.feasibility_preprocessing import preprocess_feasibility
from utils.route_preprocessing import preprocess_routes
from utils.heuristics import max_vessels_heuristic


# Class which can have attributes set.
class expando(object):
    pass


# Master problem
class Master_problem:
    def __init__(
        self,
        INSTANCE,
        NUM_WEEKS=1,
        NUM_SCENARIOS=27,
        NUM_VESSELS=1,
        MAX_PORT_VISITS=1,
        DRAW=False,
        WEEKLY_ROUTING=False,
        DISCOUNT_FACTOR=0.95,
        BENDERS_GAP=0.001,
        MAX_ITERS=1000,
        TIME_LIMIT=36000,
        HEURISTICS=False,
        VESSEL_CHANGES=1,
        warm_start=True,
    ):
        self.INSTANCE = INSTANCE
        self.iter = 0
        self.warm_start = warm_start
        self.data = expando()
        self.variables = expando()
        self.constraints = expando()
        self.results = expando()
        self._load_data(
            INSTANCE,
            TIME_LIMIT,
            MAX_ITERS,
            HEURISTICS,
            NUM_WEEKS,
            NUM_SCENARIOS,
            NUM_VESSELS,
            MAX_PORT_VISITS,
            DRAW,
            WEEKLY_ROUTING,
            DISCOUNT_FACTOR,
            BENDERS_GAP,
            VESSEL_CHANGES,
        )
        self._build_model()

    ###
    #   Loading functions
    ###

    def _load_data(
        self,
        INSTANCE,
        TIME_LIMIT,
        MAX_ITERS,
        HEURISTICS,
        NUM_WEEKS,
        NUM_SCENARIOS,
        NUM_VESSELS,
        MAX_PORT_VISITS,
        DRAW,
        WEEKLY_ROUTING,
        DISCOUNT_FACTOR,
        BENDERS_GAP,
        VESSEL_CHANGES,
    ):
        # DIRECT INPUT
        self.data.NUM_WEEKS = NUM_WEEKS
        self.data.TIME_LIMIT = TIME_LIMIT
        self.data.MAX_ITERS = MAX_ITERS
        self.data.HEURISTICS = HEURISTICS
        self.data.NUM_SCENARIOS = NUM_SCENARIOS
        self.data.NUM_VESSELS = NUM_VESSELS
        self.data.MAX_PORT_VISITS = MAX_PORT_VISITS
        self.data.DRAW = DRAW
        self.data.WEEKLY_ROUTING = WEEKLY_ROUTING
        self.data.DISCOUNT_FACTOR = DISCOUNT_FACTOR
        self.data.BENDERS_GAP = BENDERS_GAP
        self.data.VESSEL_CHANGES = VESSEL_CHANGES

        # INSTANCE DATA
        # Input data
        CUSTOMERS = self.data.CUSTOMERS = pd.read_csv(
            f"Data/Instances/{INSTANCE}/Input_data/Customer_Data.csv", index_col=0
        )
        CUSTOMERS["Scenarios"] = CUSTOMERS["Scenarios"].apply(str)
        PORTS = self.data.PORTS = pd.read_csv(
            f"Data/Instances/{INSTANCE}/Input_data/Port_Data.csv", index_col=0
        )
        VESSELS = self.data.VESSELS = pd.read_csv(
            f"Data/Instances/{INSTANCE}/Input_data/Vessel_Data.csv", index_col=0
        )[:NUM_VESSELS]
        PORT_CUSTOMER_DISTANCES = self.data.PORT_CUSTOMER_DISTANCES = pd.read_csv(
            f"Data/Instances/{INSTANCE}/Input_data/Port_Customer_Distances.csv",
            index_col=0,
        )
        self.data.TIMEPERIOD_DURATION = pd.read_csv(
            f"Data/Instances/{INSTANCE}/Input_data/Timeperiod_Duration.csv", index_col=0
        )
        # Nodes and scenario data
        NODES_IN_SCENARIO = self.data.NODES_IN_SCENARIO = pd.read_csv(
            f"Data/Nodes_and_scenarios/Nodes_in_Scenario_{NUM_SCENARIOS}.csv",
            index_col=0,
        )
        YEAR_OF_NODE = self.data.YEAR_OF_NODE = pd.read_csv(
            f"Data/Nodes_and_scenarios/Year_of_Node_{NUM_SCENARIOS}.csv", index_col=0
        )
        CO2_SCALE_FACTOR = self.data.CO2_SCALE_FACTOR = pd.read_csv(
            f"Data/Nodes_and_scenarios/CO2_Scale_Factor_{NUM_SCENARIOS}.csv",
            index_col=0,
        )
        self.data.SCENARIOYEAR_GROWTH_FACTOR = pd.read_csv(
            f"Data/Nodes_and_scenarios/ScenarioYear_Growth_Factor_{NUM_SCENARIOS}.csv",
            index_col=0,
        )
        self.data.PROB_SCENARIO = pd.read_csv(
            f"Data/Nodes_and_scenarios/Prob_Scenario_{NUM_SCENARIOS}.csv", index_col=0
        )
        # Preprocess additional data
        self._preprocess_routing()
        # Generated data
        ROUTES = self.data.ROUTES = pd.read_csv(
            f"Data/Instances/{INSTANCE}/Generated_data/Routes.csv", index_col=0
        )
        ROUTE_FEASIBILITY = self.data.ROUTE_FEASIBILITY = pd.read_csv(
            f"Data/Instances/{INSTANCE}/Generated_data/Route_Feasibility.csv",
            index_col=0,
        )
        PORT_CUSTOMER_FEASIBILITY = self.data.PORT_CUSTOMER_FEASIBILITY = pd.read_csv(
            f"Data/Instances/{INSTANCE}/Generated_data/Port_Customer_Feasibility.csv",
            index_col=0,
        )
        ROUTE_SAILING_COST = self.data.ROUTE_SAILING_COST = pd.read_csv(
            f"Data/Instances/{INSTANCE}/Generated_data/Route_Sailing_Cost.csv",
            index_col=0,
        )
        self.data.ROUTE_SAILING_TIME = pd.read_csv(
            f"Data/Instances/{INSTANCE}/Generated_data/Route_Sailing_Time.csv",
            index_col=0,
        )
        # Set sizes
        NUM_CUSTOMERS = self.data.NUM_CUSTOMERS = CUSTOMERS.shape[0]
        NUM_YEARS = self.data.NUM_YEARS = 20  # This is never changed
        NUM_PORTS = self.data.NUM_PORTS = PORTS.shape[0]
        NUM_NODES = self.data.NUM_NODES = YEAR_OF_NODE.shape[1]
        NUM_ROUTES = self.data.NUM_ROUTES = ROUTES.shape[1]
        # Generate sets and subsets
        P = self.data.P = np.arange(NUM_PORTS)
        V = self.data.V = np.arange(NUM_VESSELS)
        R = self.data.R = np.arange(NUM_ROUTES)
        N = self.data.N = np.arange(NUM_NODES)
        K = self.data.K = np.arange(NUM_CUSTOMERS)
        T = self.data.T = np.arange(NUM_YEARS)
        S = self.data.S = np.arange(NUM_SCENARIOS)
        R_v = self.data.R_v = route_vessel_set_generator(ROUTE_FEASIBILITY, V)
        self.data.BETA = beta_set_generator(YEAR_OF_NODE, NUM_NODES, NUM_YEARS)
        self.data.A_r = arc_set_generator(ROUTES, R)
        self.data.P_r = port_route_set_generator(ROUTES, P, R)
        self.data.W = np.arange(NUM_WEEKS)
        self.data.N_s = [
            NODES_IN_SCENARIO.iloc[:, c].to_numpy()
            for c in range(NODES_IN_SCENARIO.shape[1])
        ]  # List of list with all the nodes for a given scenario
        self.data.S_n = scenario_node_set_generator(NODES_IN_SCENARIO, N, S)
        self.data.NP_n = parent_node_set_generator(NODES_IN_SCENARIO, N)
        self.data.R_vi = route_vessel_port_set_generator(ROUTES, R_v, V, P)
        self.data.P_k = port_customer_set_generator(PORT_CUSTOMER_FEASIBILITY, P, K)
        self.data.N_s = [
            NODES_IN_SCENARIO.iloc[:, c].to_numpy()
            for c in range(NODES_IN_SCENARIO.shape[1])
        ]  # List of list with all the nodes for a given scenario
        self.data.K_i = [
            [int(x) for x in PORT_CUSTOMER_FEASIBILITY.iloc[:, i].dropna().to_numpy()]
            for i in P
        ]  # List of list with all serviceable customer k for a port i
        self.data.T_n = year_node_set_generator(
            NODES_IN_SCENARIO, YEAR_OF_NODE, NUM_YEARS, N
        )
        # Generate cost data
        self.data.VESSEL_INVESTMENT = vessel_investment(
            VESSELS, YEAR_OF_NODE, V, N, DISCOUNT_FACTOR
        )
        self.data.PORT_INVESTMENT = port_investment(
            PORTS[["Investment"]], YEAR_OF_NODE, P, N, DISCOUNT_FACTOR
        )
        self.data.SAILING_COST = sailing_cost(
            ROUTE_SAILING_COST, V, R, T, DISCOUNT_FACTOR
        )
        self.data.TRUCK_COST = truck_cost(
            PORT_CUSTOMER_DISTANCES, CUSTOMERS, PORTS, CO2_SCALE_FACTOR, P, K, T, S
        )
        self.data.PORT_HANDLING = port_handling_cost(
            PORTS[["Handling cost"]], P, T, DISCOUNT_FACTOR
        )
        # Make demand data
        self._make_demand_data()

        # GENERAL DATA
        self.data.warm_start_solve_time = 0
        self.data.mp_solve_time = []
        self.data.sp_solve_time = []
        self.data.upper_bounds = [GRB.INFINITY]
        self.data.lower_bounds = [-GRB.INFINITY]
        self.data.phis = [[] for _ in N]
        self.data.vessels = {}
        self.data.ports = {}
        self.sp_data = {n: expando() for n in N}
        for n in N:
            for v in V:
                self.data.vessels[(v, n)] = []
            for i in P:
                self.data.ports[(i, n)] = []

        return

    def _make_demand_data(self):

        CUSTOMERS = self.data.CUSTOMERS
        # Preallocate the 4D demand matrices
        self.data.DELIVERY = np.zeros(
            self.data.NUM_CUSTOMERS
            * self.data.NUM_YEARS
            * self.data.NUM_WEEKS
            * self.data.NUM_SCENARIOS
        ).reshape(
            self.data.NUM_CUSTOMERS,
            self.data.NUM_YEARS,
            self.data.NUM_WEEKS,
            self.data.NUM_SCENARIOS,
        )
        self.data.PICKUP = self.data.DELIVERY.copy()

        # Read the WVF
        WEEKLY_VARIATION_FACTOR = np.fromfile(
            f"Data/Instances/{self.INSTANCE}/Generated_data/Weekly_Variation_Factor.csv"
        ).reshape(self.data.NUM_CUSTOMERS, 52, self.data.NUM_YEARS)

        # Fill the matrices
        for k in self.data.K:
            delivery_split = CUSTOMERS["Delivery"][k]
            pickup_split = CUSTOMERS["Pickup"][k]
            start_year_of_customer = CUSTOMERS["Start_year"][k]
            end_year_of_customer = CUSTOMERS["End_year"][k]
            scenarios_with_customer = [
                int(s) for s in CUSTOMERS["Scenarios"][k].split(sep=",")
            ]  # a list of ints. Can take [-1] for all scenarios, a specified scenarios ex [2] or several ex [2,4,8]

            for w in self.data.W:
                for t in self.data.T:
                    # Decide whether to draw random or once drawed weeks
                    if self.data.DRAW == True:
                        customer_type = CUSTOMERS.iloc[k, 0]
                        wvf = draw_weekly_demand(1, customer_type)
                    else:
                        wvf = WEEKLY_VARIATION_FACTOR[k, w, t]

                    for s in self.data.S:
                        if (
                            start_year_of_customer == 0
                        ):  # Customers part of the network from the beginning
                            if (
                                end_year_of_customer == self.data.NUM_YEARS
                                or t < end_year_of_customer
                            ):  # Customers part of the network for the entire periode or not yet left
                                include_customer = 1
                            else:  # Customers having left the network
                                include_customer = int(
                                    not (
                                        scenarios_with_customer[0] in [-1, s]
                                        or s in scenarios_with_customer
                                    )
                                )
                        elif (
                            t >= start_year_of_customer
                        ):  # Customers joining the netwok after a given period of time
                            include_customer = int(
                                scenarios_with_customer[0] in [-1, s]
                                or s in scenarios_with_customer
                            )
                        else:
                            include_customer = 0

                        baseline_delivery = delivery_split * np.prod(
                            self.data.SCENARIOYEAR_GROWTH_FACTOR.iloc[s, : t + 1]
                        )
                        self.data.DELIVERY[k, t, w, s] = (
                            int(baseline_delivery * wvf) * include_customer
                        )

                        baseline_pickup = pickup_split * np.prod(
                            self.data.SCENARIOYEAR_GROWTH_FACTOR.iloc[s, : t + 1]
                        )
                        self.data.PICKUP[k, t, w, s] = (
                            int(baseline_pickup * wvf) * include_customer
                        )

        return

    def _preprocess_routing(self):
        preprocess_feasibility(self.INSTANCE)
        preprocess_routes(self.INSTANCE, self.data.MAX_PORT_VISITS)
        return

    ###
    #   Model Building
    ###
    def _build_model(self):
        self.m = gp.Model(env=gp.Env(params={"OutputFlag": 0, "LazyConstraints": 1}))
        # self.m.setParam("LazyConstraints", 1)  # Allow the use of lazy constraints
        # self.m.setParam("OutputFlag", 0)  # Suppress default output
        self._build_variables()
        self._build_constraints()
        self._build_objective()
        self.m.update()

        return

    def _build_variables(self):
        # Fetch data
        m = self.m

        V = self.data.V
        P = self.data.P
        N = self.data.N

        self.variables.vessels = m.addVars(
            product(V, N), vtype=GRB.INTEGER, name="Vessel investment"
        )
        self.variables.ports = m.addVars(
            product(P, N), vtype=GRB.BINARY, name="Port investment"
        )
        self.variables.phi = m.addVars(N, vtype=GRB.CONTINUOUS, name="Phi")

        m.update()

        return

    def _build_constraints(self):
        # Only open a port in one of the years in a scenarios
        self.constraints.max_ports_in_scenario = self.m.addConstrs(
            gp.quicksum(self.variables.ports[(i, n)] for n in self.data.N_s[s]) <= 1
            for i in self.data.P
            for s in self.data.S
        )
        if self.data.HEURISTICS:
            max_vessels = max_vessels_heuristic(self)
            self.constraints.max_vessels_heuristic = self.m.addConstrs(
                gp.quicksum(self.variables.vessels[(v, n)] for n in self.data.N_s[s])
                <= max_vessels[v]
                for v in self.data.V
                for s in self.data.S
            )
            self.constraints.limit_vessel_decrease = self.m.addConstrs(
                self.variables.vessels[(v, n)] <= 0
                for v in self.data.V
                for n in self.data.N
            )
            self.constraints.limit_vessel_increase = self.m.addConstrs(
                -self.variables.vessels[(v, n)] <= 0
                for v in self.data.V
                for n in self.data.N
            )
        return

    def _build_objective(self):

        # Fetch sets
        V = self.data.V
        P = self.data.P
        N = self.data.N
        N_s = self.data.N_s
        S = self.data.S

        # Fetch data
        PROB_SCENARIO = self.data.PROB_SCENARIO
        VESSEL_INVESTMENT = self.data.VESSEL_INVESTMENT
        PORT_INVESTMENT = self.data.PORT_INVESTMENT

        # Fetch variables
        vessels = self.variables.vessels
        ports = self.variables.ports
        # Make the objective function for the master problem
        MP_obj_func = gp.quicksum(
            PROB_SCENARIO.iloc[0, s]
            * gp.quicksum(
                gp.quicksum(VESSEL_INVESTMENT[v, n] * vessels[(v, n)] for v in V)
                + gp.quicksum(PORT_INVESTMENT[i, n] * ports[(i, n)] for i in P)
                for n in N_s[s]
            )
            for s in S
        ) + gp.quicksum(self.variables.phi[n] for n in N)

        self.m.setObjective(MP_obj_func, gp.GRB.MINIMIZE)

        return

    def _make_subproblems(self) -> None:
        self.subproblems = {n: Subproblem(NODE=n, mp=self) for n in self.data.N}
        return

    def _check_termination(self, run_start):
        terminate = False
        try:
            lb = self.data.lower_bounds[
                -1
            ]  # lb increasing in each iteration, lb max is the last element
            ub = min(
                self.data.upper_bounds
            )  # ub is the lowest ub found up until the current iteration.
            gap = (ub - lb) / lb * 100
            print(
                f"BOUNDS: UB = {int(ub/1e6)} | LB = {int(lb/1e6)} | Gap = {round(gap,2)} %"
            )
        except:
            print(f"Iteration {self.iter}. Bounds not applicable")

        # 4.1 Relative gap < 1%
        if ub <= lb * 1.0001:
            print(f"**OPTIMAL SOLUTION FOUND: {int(ub*1e-6)}**")
            terminate = True
            # 4.3 Number of iterations
        elif self.iter + 1 > self.data.MAX_ITERS:
            print(f"**MAX ITERATIONS REACHED {self.data.MAX_ITERS}**")
            terminate = True
        elif time() - run_start > self.data.TIME_LIMIT:
            print(f"**TIME LIMIT REACHED {self.data.TIME_LIMIT}**")
            terminate = True

        return terminate

    def _warm_start(self):
        print(f"\n>>>ITERATION {self.iter} | warm-start")
        t0 = time()

        self.fp = Full_problem(self, SCENARIOS=[4])
        self.fp.solve()
        self._save_vars(self.fp)

        for n in self.data.N:
            self.subproblems[n]._update_fixed_vars()
            self.subproblems[n].solve()
        self._add_cut(self.data.N)
        if self.data.HEURISTICS:
            self._update_vessel_changes(self.fp)
        self.warm_start = False

        t1 = time()
        self.data.warm_start_solve_time = t1 - t0
        self.iter += 1

        return

    def _add_cut(self, N):
        m = self.m

        # Imports sets and other necessary data
        V = self.data.V
        P = self.data.P
        NP_n = self.data.NP_n

        for n in N:
            lhs = self.variables.phi[n]
            z_sub = self.subproblems[n].data.obj_vals[-1]
            sens_vessels = self.subproblems[n].data.sens_vessels[-1]
            sens_ports = self.subproblems[n].data.sens_ports[-1]
            rhs = (
                z_sub
                + gp.quicksum(
                    sens_vessels[v]
                    * gp.quicksum(
                        self.variables.vessels[(v, m)] - self.data.vessels[(v, m)][-1]
                        for m in NP_n[n]
                    )
                    for v in V
                )
                + gp.quicksum(
                    sens_ports[i]
                    * gp.quicksum(
                        self.variables.ports[(i, m)] - self.data.ports[(i, m)][-1]
                        for m in NP_n[n]
                    )
                    for i in P[1:]
                )
            )
            m.addConstr(lhs >= rhs)

        return

    ###
    #
    ####
    def _update_bounds(self, N_4bounds=None):
        m = self.m
        N = self.data.N

        if N_4bounds is None:
            N_4bounds = [-1 for _ in N]

        # Fetch the current value of the master problem and the artificial variable phi
        z_master = m.ObjVal
        phi_val = sum([self.variables.phi[n].x for n in N])
        z_sub_total = sum([self.subproblems[n].data.obj_vals[N_4bounds[n]] for n in N])

        # The best upper bound is the best incumbent with phi replaced by the sub problems' actual cost
        ub = z_master - phi_val + z_sub_total

        # The best lower bound is the current bestbound,
        # This will equal z_master at optimality
        lb = z_master

        self.data.upper_bounds.append(ub)
        self.data.lower_bounds.append(lb)

        return

    def _save_vars(self, model=None):
        if model is None:
            for n in self.data.N:
                self.data.phis[n].append(self.variables.phi[n].x)
                for v in self.data.V:
                    self.data.vessels[(v, n)].append(self.variables.vessels[(v, n)].x)
                for i in self.data.P:
                    self.data.ports[(i, n)].append(self.variables.ports[(i, n)].x)
        else:
            for n_solved in self.data.N_s[model.data.S[0]]:
                N = get_same_year_nodes(n_solved, self.data.N, self.data.YEAR_OF_NODE)
                for n in N:
                    for v in self.data.V:
                        self.data.vessels[(v, n)].append(
                            model.variables.vessels[(v, n_solved)].x
                        )
                    for i in self.data.P:
                        self.data.ports[(i, n)].append(
                            model.variables.ports[(i, n_solved)].x
                        )

        return

    def _update_vessel_changes(self, model=None):
        if model is None:
            for n in self.data.N:
                for v in self.data.V:
                    vessel_val = self.variables.vessels[(v, n)].x
                    self.constraints.limit_vessel_decrease[(v, n)].rhs = (
                        vessel_val + self.data.VESSEL_CHANGES
                    )
                    self.constraints.limit_vessel_increase[(v, n)].rhs = (
                        -vessel_val + self.data.VESSEL_CHANGES
                    )
        else:
            for n_solved in self.data.N_s[model.data.S[0]]:
                N = get_same_year_nodes(n_solved, self.data.N, self.data.YEAR_OF_NODE)
                for n in N:
                    for v in self.data.V:
                        vessel_val = model.variables.vessels[(v, n_solved)].x
                        self.constraints.limit_vessel_decrease[(v, n)].rhs = (
                            vessel_val + self.data.VESSEL_CHANGES
                        )
                        self.constraints.limit_vessel_increase[(v, n)].rhs = (
                            -vessel_val + self.data.VESSEL_CHANGES
                        )

        return
