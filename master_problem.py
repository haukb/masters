# Import libraries
import gurobipy as gp
from gurobipy import GRB
from numpy.core.fromnumeric import shape
import pandas as pd
import numpy as np
from itertools import product

from time import time
# Import other classes
from subproblem import Subproblem
from full_problem import Full_problem

# Import helper functions
from special_set_generators import beta_set_generator, arc_set_generator, port_route_set_generator, route_vessel_set_generator, route_vessel_port_set_generator, port_customer_set_generator, scenario_node_set_generator, year_node_set_generator, parent_node_set_generator
from cost_generators import vessel_investment, port_investment, sailing_cost, truck_cost, port_handling_cost
from short_term_uncertainty import draw_weekly_demand
from misc_functions import get_same_year_nodes, nodes_with_new_investments
from feasibility_preprocessing import preprocess_feasibility
from route_preprocessing import preprocess_routes

# Class which can have attributes set.
class expando(object):
    pass

# Master problem
class Master_problem:
    def __init__(self, INSTANCE, NUM_WEEKS = 1, NUM_SCENARIOS = 27, NUM_VESSELS = 1, MAX_PORT_VISITS = 3, DRAW = False, WEEKLY_ROUTING = False, DISCOUNT_FACTOR = 1, BENDERS_GAP=0.01, MAX_ITERS=10, warm_start = True):
        self.INSTANCE = INSTANCE
        self.MAX_ITERS = MAX_ITERS
        self.iter = 0
        self.warm_start = warm_start
        self.data = expando()
        self.variables = expando()
        self.constraints = expando()
        self.results = expando()
        self._load_data(INSTANCE, NUM_WEEKS, NUM_SCENARIOS, NUM_VESSELS, MAX_PORT_VISITS, DRAW, WEEKLY_ROUTING, DISCOUNT_FACTOR, BENDERS_GAP)
        self._build_model()
    ###
    #   Loading functions
    ###

    def _load_data(self, INSTANCE, NUM_WEEKS, NUM_SCENARIOS, NUM_VESSELS, MAX_PORT_VISITS, DRAW, WEEKLY_ROUTING, DISCOUNT_FACTOR, BENDERS_GAP):
        #DIRECT INPUT
        self.data.NUM_WEEKS = NUM_WEEKS
        self.data.NUM_SCENARIOS = NUM_SCENARIOS
        self.data.NUM_VESSELS = NUM_VESSELS
        self.data.MAX_PORT_VISITS = MAX_PORT_VISITS
        self.data.DRAW = DRAW
        self.data.WEEKLY_ROUTING = WEEKLY_ROUTING
        self.data.DISCOUNT_FACTOR = DISCOUNT_FACTOR
        self.data.BENDERS_GAP = BENDERS_GAP

        #INSTANCE DATA
        #Input data
        CUSTOMERS = self.data.CUSTOMERS = pd.read_csv(f'Data/Instances/{INSTANCE}/Input_data/Customer_Data.csv', index_col=0)
        CUSTOMERS['Scenarios'] = CUSTOMERS['Scenarios'].apply(str)
        PORTS = self.data.PORTS = pd.read_csv(f'Data/Instances/{INSTANCE}/Input_data/Port_Data.csv', index_col=0)
        VESSELS = self.data.VESSELS = pd.read_csv(f'Data/Instances/{INSTANCE}/Input_data/Vessel_Data.csv', index_col=0)[:NUM_VESSELS]
        PORT_CUSTOMER_DISTANCES = self.data.PORT_CUSTOMER_DISTANCES = pd.read_csv(f'Data/Instances/{INSTANCE}/Input_data/Port_Customer_Distances.csv', index_col=0)
        self.data.TIMEPERIOD_DURATION = pd.read_csv(f'Data/Instances/{INSTANCE}/Input_data/Timeperiod_Duration.csv', index_col=0)
        #Nodes and scenario data
        NODES_IN_SCENARIO = self.data.NODES_IN_SCENARIO = pd.read_csv(f'Data/Nodes_and_scenarios/Nodes_in_Scenario_{NUM_SCENARIOS}.csv', index_col=0)
        YEAR_OF_NODE = self.data.YEAR_OF_NODE = pd.read_csv(f'Data/Nodes_and_scenarios/Year_of_Node_{NUM_SCENARIOS}.csv', index_col=0)
        CO2_SCALE_FACTOR = self.data.CO2_SCALE_FACTOR = pd.read_csv(f'Data/Nodes_and_scenarios/CO2_Scale_Factor_{NUM_SCENARIOS}.csv', index_col=0)
        self.data.SCENARIOYEAR_GROWTH_FACTOR = pd.read_csv(f'Data/Nodes_and_scenarios/ScenarioYear_Growth_Factor_{NUM_SCENARIOS}.csv', index_col=0)
        self.data.PROB_SCENARIO = pd.read_csv(f'Data/Nodes_and_scenarios/Prob_Scenario_{NUM_SCENARIOS}.csv', index_col=0)
        #Preprocess additional data
        self._preprocess_routing()
        #Generated data
        ROUTES = self.data.ROUTES = pd.read_csv(f'Data/Instances/{INSTANCE}/Generated_data/Routes.csv', index_col=0)
        ROUTE_FEASIBILITY = self.data.ROUTE_FEASIBILITY = pd.read_csv(f'Data/Instances/{INSTANCE}/Generated_data/Route_Feasibility.csv', index_col=0)
        PORT_CUSTOMER_FEASIBILITY = self.data.PORT_CUSTOMER_FEASIBILITY = pd.read_csv(f'Data/Instances/{INSTANCE}/Generated_data/Port_Customer_Feasibility.csv', index_col=0)
        ROUTE_SAILING_COST = self.data.ROUTE_SAILING_COST = pd.read_csv(f'Data/Instances/{INSTANCE}/Generated_data/Route_Sailing_Cost.csv', index_col=0)
        self.data.ROUTE_SAILING_TIME = pd.read_csv(f'Data/Instances/{INSTANCE}/Generated_data/Route_Sailing_Time.csv', index_col=0)
        #Set sizes
        NUM_CUSTOMERS = self.data.NUM_CUSTOMERS = CUSTOMERS.shape[0]
        NUM_YEARS = self.data.NUM_YEARS = 20 #This is never changed
        NUM_PORTS = self.data.NUM_PORTS = PORTS.shape[0]
        NUM_NODES = self.data.NUM_NODES = YEAR_OF_NODE.shape[1]
        NUM_ROUTES = self.data.NUM_ROUTES = ROUTES.shape[1]
        #Generate sets and subsets
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
        self.data.N_s = [NODES_IN_SCENARIO.iloc[:,c].to_numpy() for c in range(NODES_IN_SCENARIO.shape[1])] #List of list with all the nodes for a given scenario
        self.data.S_n = scenario_node_set_generator(NODES_IN_SCENARIO, N, S)
        self.data.NP_n = parent_node_set_generator(NODES_IN_SCENARIO, N)
        self.data.R_vi = route_vessel_port_set_generator(ROUTES, R_v, V, P)
        self.data.P_k = port_customer_set_generator(PORT_CUSTOMER_FEASIBILITY, P, K)
        self.data.N_s = [NODES_IN_SCENARIO.iloc[:,c].to_numpy() for c in range(NODES_IN_SCENARIO.shape[1])] #List of list with all the nodes for a given scenario
        self.data.K_i = [[int(x) for x in PORT_CUSTOMER_FEASIBILITY.iloc[:,i].dropna().to_numpy()] for i in P] #List of list with all serviceable customer k for a port i
        self.data.T_n = year_node_set_generator(NODES_IN_SCENARIO, YEAR_OF_NODE, NUM_YEARS, N)
        #Generate cost data
        self.data.VESSEL_INVESTMENT = vessel_investment(VESSELS,YEAR_OF_NODE, V, N, DISCOUNT_FACTOR)
        self.data.PORT_INVESTMENT = port_investment(PORTS[['Investment']],YEAR_OF_NODE, P, N, DISCOUNT_FACTOR)
        self.data.SAILING_COST = sailing_cost(ROUTE_SAILING_COST, V, R, T, DISCOUNT_FACTOR)
        self.data.TRUCK_COST = truck_cost(PORT_CUSTOMER_DISTANCES, CUSTOMERS, PORTS, CO2_SCALE_FACTOR, P, K, T, S)
        self.data.PORT_HANDLING = port_handling_cost(PORTS[['Handling cost']], P, T, DISCOUNT_FACTOR)
        #Make demand data
        self._make_demand_data()

        #GENERAL DATA
        self.data.cutlist = []
        self.data.upper_bounds = [GRB.INFINITY]
        self.data.lower_bounds = [-GRB.INFINITY]
        self.data.lambdas = {}
        self.data.ub = gp.GRB.INFINITY
        self.data.lb = -gp.GRB.INFINITY
        self.data.phis = [[] for n in N]
        self.data.vessels = {}
        for v in V:
            for n in N:
                self.data.vessels[(v,n)] = []
        self.data.ports = {}
        for i in P:
            for n in N:
                self.data.ports[(i,n)] = []

        return

    def _make_demand_data(self):
    
        CUSTOMERS = self.data.CUSTOMERS
        #Preallocate the 4D demand matrices 
        self.data.DELIVERY = np.zeros(self.data.NUM_CUSTOMERS*self.data.NUM_YEARS*self.data.NUM_WEEKS*self.data.NUM_SCENARIOS).reshape(self.data.NUM_CUSTOMERS,self.data.NUM_YEARS,self.data.NUM_WEEKS,self.data.NUM_SCENARIOS)
        self.data.PICKUP = self.data.DELIVERY.copy()

        #Read the WVF
        WEEKLY_VARIATION_FACTOR = np.fromfile(f'Data/Instances/{self.INSTANCE}/Generated_data/Weekly_Variation_Factor.csv').reshape(self.data.NUM_CUSTOMERS, 52, self.data.NUM_YEARS)

        #Fill the matrices
        for k in self.data.K:
            delivery_split = CUSTOMERS['Delivery'][k]
            pickup_split = CUSTOMERS['Pickup'][k]
            start_year_of_customer = CUSTOMERS['Start_year'][k]
            end_year_of_customer = CUSTOMERS['End_year'][k]
            scenarios_with_customer = [int(s) for s in CUSTOMERS['Scenarios'][k].split(sep=',')] # a list of ints. Can take [-1] for all scenarios, a specified scenarios ex [2] or several ex [2,4,8] 
            
            for w in self.data.W:
                for t in self.data.T:
                    #Decide whether to draw random or once drawed weeks
                    if self.data.DRAW == True: 
                        customer_type = CUSTOMERS.iloc[k,0]
                        wvf = draw_weekly_demand(1, customer_type)
                    else: 
                        wvf = WEEKLY_VARIATION_FACTOR[k, w, t]
            
                    for s in self.data.S:
                        if start_year_of_customer == 0: #Customers part of the network from the beginning
                            if end_year_of_customer == self.data.NUM_YEARS or t<end_year_of_customer: #Customers part of the network for the entire periode or not yet left
                                include_customer = 1
                            else: #Customers having left the network
                                include_customer = int(not(scenarios_with_customer[0] in [-1,s] or s in scenarios_with_customer))
                        elif t>=start_year_of_customer: #Customers joining the netwok after a given period of time
                            include_customer = int(scenarios_with_customer[0] in [-1,s] or s in scenarios_with_customer) 
                        else: 
                            include_customer = 0

                        baseline_delivery = delivery_split*np.prod(self.data.SCENARIOYEAR_GROWTH_FACTOR.iloc[s,:t+1])
                        self.data.DELIVERY[k,t,w,s] = int(baseline_delivery*wvf)*include_customer
                            
                        baseline_pickup = pickup_split*np.prod(self.data.SCENARIOYEAR_GROWTH_FACTOR.iloc[s,:t+1])
                        self.data.PICKUP[k,t,w,s] = int(baseline_pickup*wvf)*include_customer

        return
    
    def _preprocess_routing(self):
        preprocess_feasibility(self.INSTANCE)
        preprocess_routes(self.INSTANCE, self.data.MAX_PORT_VISITS)
        return

    ###
    #   Model Building
    ###
    def _build_model(self):
        self.m = gp.Model()
        self.m.setParam('LazyConstraints', 1) #Allow the use of lazy constraints
        self.m.setParam('OutputFlag', 0) #Suppress default output
        self._build_variables()
        self._build_constraints()
        self._build_objective()
        self.m.update()

    def _build_variables(self):
        #Fetch data
        m = self.m

        V = self.data.V
        P = self.data.P
        N = self.data.N 

        self.variables.vessels = m.addVars(product(V,N), vtype=GRB.INTEGER, name="Vessel investment")
        self.variables.ports = m.addVars(product(P,N), vtype=GRB.BINARY, name="Port investment")
        self.variables.phi = m.addVars(N, vtype = GRB.CONTINUOUS, name = "Phi")

        m.update()

        return

    def _build_constraints(self):
        #Only open port in one time, to hinder "double" opening for dual values in several nodes
        self.constraints.max_ports_in_scenario = self.m.addConstrs(gp.quicksum(self.variables.ports[(i,n)] for n in self.data.N_s[s]) <= 1 for i in self.data.P for s in self.data.S)
        self.constraints.max_number_of_vessels = self.m.addConstrs(gp.quicksum(self.variables.vessels[(v,n)] for n in self.data.N_s[s]) <= 10 for v in self.data.V for s in self.data.S)
        self.constraints.limit_vessel_decrease = self.m.addConstrs(self.variables.vessels[(v,n)] <= 0 for v in self.data.V for n in self.data.N)
        self.constraints.limit_vessel_increase = self.m.addConstrs(-self.variables.vessels[(v,n)] <= 0 for v in self.data.V for n in self.data.N)
        return

    def _build_objective(self):

        #Fetch sets
        V = self.data.V
        P = self.data.P
        N = self.data.N
        N_s = self.data.N_s
        S = self.data.S

        #Fetch data
        PROB_SCENARIO = self.data.PROB_SCENARIO
        VESSEL_INVESTMENT = self.data.VESSEL_INVESTMENT
        PORT_INVESTMENT = self.data.PORT_INVESTMENT

        #Fetch variables
        vessels = self.variables.vessels
        ports = self.variables.ports
        #Make the objective function for the master problem
        MP_obj_func = gp.quicksum(PROB_SCENARIO.iloc[0,s]*
        gp.quicksum(gp.quicksum(VESSEL_INVESTMENT[v,n]*vessels[(v, n)] for v in V) + gp.quicksum(PORT_INVESTMENT[i,n]*ports[(i, n)] for i in P) 
        for n in N_s[s]) for s in S) + gp.quicksum(self.variables.phi[n] for n in N)

        self.m.setObjective(MP_obj_func, gp.GRB.MINIMIZE)

        return
 
    def solve(self):
        mp = self
        m = self.m

        # Only build subproblems if they don't exist or a rebuild is forced.
        if not hasattr(self, 'subproblems'):# or force_submodel_rebuild:
            self.subproblems = {n: Subproblem(NODE=n, mp=self) for n in self.data.N}
        
        #Warm start algorithm with solution from deteministic problem
        if self.warm_start:
            self._warm_start()

        while True: 
            mp.iter += 1
            # 1. Solve master problem and save variables
            t0 = time()
            m.optimize()
            mp._save_vars()
            t1 = time()
            print(f'>>>Time spent solving MP: {round(t1-t0,3)}')

            # 2. Solve subproblems
            N_changed = nodes_with_new_investments(mp.data.vessels, mp.data.ports, mp.data.V, mp.data.P, mp.data.N, mp.data.NP_n)
            t0 = time()
            for n in N_changed: #OLD replace N_changed with mp.data.N 
                sp = mp.subproblems[n]
                sp.update_fixed_vars()
                sp.solve()
            t1 = time()
            print(f'>>Time spent solving SP: {round(t1-t0,3)}')

             # 3. Update the bounds on the mp
            mp._update_bounds()

            # 4. Check termination criterias: Relative gap, Absolute gap & Number of iterations 
            try: 
                lb = mp.data.lower_bounds[-1] #lb increasing in each iteration, lb max is the last element
                ub = min(mp.data.upper_bounds) #ub is the lowest ub found up until the current iteration.
                gap = (ub - lb)  / lb * 100
                print(f'> Iteration {mp.iter}. UB: {int(ub/1e6)} | LB: {int(lb/1e6)} | Gap: {round(gap,2)} %')
            except:
                print(f'> Iteration {mp.iter}. Bounds not applicable')

                # 4.1 Relative gap < 1%
            if ub <= lb + 0.001*lb:
                print(f'**OPTIMAL SOLUTION FOUND: {mp.data.upper_bounds[-1]}**')
                break
                # 4.2 Number of iterations
            elif mp.iter > mp.MAX_ITERS:
                print(f'**MAX ITERATIONS REACHED {mp.MAX_ITERS}**')
                break

            # 5. Add a cut to the mp and update the allowed vessel changes
            #mp._add_unicut()
            mp._add_multicut(N_changed)
            mp._update_vessel_changes()

        return
    
    def _warm_start(self):

        self.fp = Full_problem(self, SCENARIOS = [4])
        self.fp.solve()
        self._save_vars(self.fp)

        for n in self.data.N:
            self.subproblems[n].update_fixed_vars() 
            self.subproblems[n].solve()
        #self._add_unicut()
        self._add_multicut(self.data.N)
        self._update_vessel_changes(self.fp)
        self.warm_start = False

        return
            
    def _add_unicut(self):
        m = self.m

        #Imports sets and other necessary data
        V = self.data.V
        P = self.data.P
        N = self.data.N
        NP_n = self.data.NP_n

        # Define dictionaries for sensitivities and objective values of the subproblems
        z_sub = dict.fromkeys(N)
        sens_ports = pd.DataFrame(data = np.zeros(shape=(len(P),len(N))))
        sens_vessels = pd.DataFrame(data = np.zeros(shape=(len(V),len(N))))

        for n in N:
            # Get the probability adjusted objective values of the subproblems 
            z_sub[n] = self.subproblems[n].m.ObjVal
            for v in V:
                sens_vessels.iloc[v,n] = self.subproblems[n].constraints.fix_vessels[(v,n)].pi
            for i in P:
                sens_ports.iloc[i,n] = self.subproblems[n].constraints.fix_ports[(i,n)].pi
        
        # Generate cut
        lhs = self.variables.phi[0]
        rhs = gp.quicksum(z_sub[n]+
        gp.quicksum(sens_vessels.iloc[v,n] * gp.quicksum(self.variables.vessels[(v,m)]-self.subproblems[m].variables.vessels_free[(v,m)].x for m in NP_n[n]) for v in V) +
        gp.quicksum(sens_ports.iloc[i,n] * (self.variables.ports[(i,n)]-self.subproblems[n].variables.ports_free[(i,n)].x) for i in P[1:])
        for n in N)

        m.addConstr(lhs >= rhs)

        return

    def _add_multicut(self, N):
        m = self.m

        #Imports sets and other necessary data
        V = self.data.V
        P = self.data.P
        NP_n = self.data.NP_n

        for n in N: 
            lhs = self.variables.phi[n]
            z_sub = self.subproblems[n].data.obj_vals[-1]
            sens_vessels = self.subproblems[n].data.sens_vessels[-1]
            sens_ports = self.subproblems[n].data.sens_ports[-1]
            rhs = (z_sub + 
            gp.quicksum(sens_vessels[v] * gp.quicksum(self.variables.vessels[(v,m)]-self.data.vessels[(v,m)][-1] for m in NP_n[n]) for v in V) + 
            gp.quicksum(sens_ports[i] * gp.quicksum(self.variables.ports[(i,m)]-self.data.ports[(i,m)][-1] for m in NP_n[n]) for i in P[1:]))
            m.addConstr(lhs >= rhs)

        return

    ###
    # 
    ####
    def _update_bounds(self):
        m = self.m

        N = self.data.N

        #Fetch the current value of the master problem and the artificial variable phi at the current MIPSOL in the callback
        z_master = m.ObjVal
        phi_val = sum([self.variables.phi[n].x for n in N])
        z_sub_total = sum([self.subproblems[n].data.obj_vals[-1] for n in N])

        # The best upper bound is the best incumbent with phi replaced by the sub problems' actual cost
        self.data.ub = z_master - phi_val + z_sub_total

        # The best lower bound is the current bestbound,
        # This will equal z_master at optimality
        self.data.lb = z_master

        self.data.upper_bounds.append(self.data.ub)
        self.data.lower_bounds.append(self.data.lb)

        return

    def _save_vars(self, model = None):
        if model == None:
            for n in self.data.N:
                self.data.phis[n].append(self.variables.phi[n].x)
                for v in self.data.V:
                    self.data.vessels[(v,n)].append(self.variables.vessels[(v,n)].x)
                for i in self.data.P:
                    self.data.ports[(i,n)].append(self.variables.ports[(i,n)].x)
        else: 
            for n_solved in self.data.N_s[model.data.S[0]]:
                N = get_same_year_nodes(n_solved, self.data.N, self.data.YEAR_OF_NODE)
                for n in N:
                    for v in self.data.V:
                        self.data.vessels[(v,n)].append(model.variables.vessels[(v,n_solved)].x)
                    for i in self.data.P:
                        self.data.ports[(i,n)].append(model.variables.ports[(i,n_solved)].x)

        return

    def _update_vessel_changes(self, model = None):
        if model == None:
            for n in self.data.N:
                for v in self.data.V:
                    vessel_val = self.variables.vessels[(v,n)].x
                    self.constraints.limit_vessel_decrease[(v,n)].rhs = vessel_val + 1
                    self.constraints.limit_vessel_increase[(v,n)].rhs = -vessel_val + 1
        else: 
            for n_solved in self.data.N_s[model.data.S[0]]:
                N = get_same_year_nodes(n_solved, self.data.N, self.data.YEAR_OF_NODE)
                for n in N:
                    for v in self.data.V:
                        vessel_val = model.variables.vessels[(v,n_solved)].x
                        self.constraints.limit_vessel_decrease[(v,n)].rhs = vessel_val + 1
                        self.constraints.limit_vessel_increase[(v,n)].rhs = -vessel_val + 1

        return
