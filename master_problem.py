# Import libraries
import gurobipy as gp
from gurobipy import GRB
from numpy.core.fromnumeric import shape
import pandas as pd
import numpy as np
from itertools import product

# Import other classes
from subproblem import Subproblem
from full_problem import Full_problem

# Import helper functions
from special_set_generators import beta_set_generator, arc_set_generator, port_route_set_generator, route_vessel_set_generator, route_vessel_port_set_generator, port_customer_set_generator, scenario_node_set_generator, year_node_set_generator
from cost_generators import vessel_investment, port_investment, sailing_cost, truck_cost, port_handling_cost
from short_term_uncertainty import draw_weekly_demand
from misc_functions import get_same_year_nodes

# Class which can have attributes set.
class expando(object):
    pass

# Master problem
class Master_problem:
    def __init__(self, INSTANCE, NUM_WEEKS = 1, NUM_SCENARIOS = 27, NUM_VESSELS = 1, MAX_PORT_VISITS = 1, DRAW = False, WEEKLY_ROUTING = False, DISCOUNT_FACTOR = 1, BENDERS_GAP=0.001, MAX_ITERS=10, hot_start = True):
        self.INSTANCE = INSTANCE
        self.MAX_ITERS = MAX_ITERS
        self.iter = 0
        self.hot_start = hot_start
        self.data = expando()
        self.variables = expando()
        self.constraints = expando()
        self.results = expando()
        self.count_loops = 0
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
        CUSTOMERS = self.data.CUSTOMERS = pd.read_csv(f'TestData/{INSTANCE}/Input_data/Customer_Data.csv', index_col=0)
        PORTS = self.data.PORTS = pd.read_csv(f'TestData/{INSTANCE}/Input_data/Port_Data.csv', index_col=0)
        VESSELS = self.data.VESSELS = pd.read_csv(f'TestData/{INSTANCE}/Input_data/Vessel_Data.csv', index_col=0)[:NUM_VESSELS]
        NODES_IN_SCENARIO = self.data.NODES_IN_SCENARIO = pd.read_csv(f'TestData/Scenarios/Nodes_in_Scenario_{NUM_SCENARIOS}.csv', index_col=0)
        YEAR_OF_NODE = self.data.YEAR_OF_NODE = pd.read_csv(f'TestData/Scenarios/Year_of_Node_{NUM_SCENARIOS}.csv', index_col=0)
        ROUTES = self.data.ROUTES = pd.read_csv(f'TestData/{INSTANCE}/Generated_data/Routes.csv', index_col=0)
        ROUTE_FEASIBILITY = self.data.ROUTE_FEASIBILITY = pd.read_csv(f'TestData/{INSTANCE}/Generated_data/Route_Feasibility.csv', index_col=0)
        PORT_CUSTOMER_FEASIBILITY = self.data.PORT_CUSTOMER_FEASIBILITY = pd.read_csv(f'TestData/{INSTANCE}/Generated_data/Port_Customer_Feasibility.csv', index_col=0)
        PORT_CUSTOMER_DISTANCES = self.data.PORT_CUSTOMER_DISTANCES = pd.read_csv(f'TestData/{INSTANCE}/Input_data/Port_Customer_Distances.csv', index_col=0)
        CO2_SCALE_FACTOR = self.data.CO2_SCALE_FACTOR = pd.read_csv(f'TestData/Scenarios/CO2_Scale_Factor_{NUM_SCENARIOS}.csv', index_col=0)
        ROUTE_SAILING_COST = self.data.ROUTE_SAILING_COST = pd.read_csv(f'TestData/{INSTANCE}/Generated_data/Route_Sailing_Cost.csv', index_col=0)
        self.data.SCENARIOYEAR_GROWTH_FACTOR = pd.read_csv(f'TestData/Scenarios/ScenarioYear_Growth_Factor_{NUM_SCENARIOS}.csv', index_col=0)
        self.data.PROB_SCENARIO = pd.read_csv(f'TestData/Scenarios/Prob_Scenario_{NUM_SCENARIOS}.csv', index_col=0)
        self.data.TIMEPERIOD_DURATION = pd.read_csv(f'TestData/{INSTANCE}/Input_data/Timeperiod_Duration.csv', index_col=0)
        self.data.ROUTE_SAILING_TIME = pd.read_csv(f'TestData/{INSTANCE}/Generated_data/Route_Sailing_Time.csv', index_col=0)
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
        self.data.R_vi = route_vessel_port_set_generator(ROUTES, R_v, V, P)
        self.data.P_k = port_customer_set_generator(PORT_CUSTOMER_FEASIBILITY, P, K)
        self.data.N_s = [NODES_IN_SCENARIO.iloc[:,c].to_numpy() for c in range(NODES_IN_SCENARIO.shape[1])] #List of list with all the nodes for a given scenario
        self.data.K_i = [[int(x) for x in PORT_CUSTOMER_FEASIBILITY.iloc[:,i].dropna().to_numpy()] for i in P] #List of list with all serviceable customer k for a port i
        self.data.S_n = scenario_node_set_generator(NODES_IN_SCENARIO, N, S)
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
        self.data.phis = []
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
        WEEKLY_VARIATION_FACTOR = np.fromfile(f'TestData/{self.INSTANCE}/Generated_data/WEEKLY_VARIATION_FACTOR.csv').reshape(self.data.NUM_CUSTOMERS, 52, self.data.NUM_YEARS)

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
        self.variables.phi = m.addVar(vtype = GRB.CONTINUOUS, name = "Phi")

        m.update()

        return

    def _build_constraints(self):
        #Only open port in one time, to hinder "double" opening for dual values in several nodes
        self.constraints.max_ports_in_scenario = self.m.addConstrs(gp.quicksum(self.variables.ports[(i,n)] for n in self.data.N_s[s]) <= 1 for i in self.data.P for s in self.data.S)
        self.constraints.max_number_of_vessels = self.m.addConstrs(gp.quicksum(self.variables.vessels[(v,n)] for n in self.data.N_s[s]) <= 10 for v in self.data.V for s in self.data.S)
        return

    def _build_objective(self):

        #Fetch sets
        V = self.data.V
        P = self.data.P
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
        #gp.quicksum(expression for a in A)
        MP_obj_func = gp.quicksum(PROB_SCENARIO.iloc[0,s]*
        gp.quicksum(gp.quicksum(VESSEL_INVESTMENT[v,n]*vessels[(v, n)] for v in V) + gp.quicksum(PORT_INVESTMENT[i,n]*ports[(i, n)] for i in P) 
        for n in N_s[s]) for s in S) + self.variables.phi

        self.m.setObjective(MP_obj_func, gp.GRB.MINIMIZE)

        return
 
    def solve(self):
        mp = self
        m = self.m

        # Only build subproblems if they don't exist or a rebuild is forced.
        if not hasattr(self, 'subproblems'):# or force_submodel_rebuild:
            self.subproblems = {n: Subproblem(self, NODE=n) for n in self.data.N}
        # Update fixed variables for subproblems and rebuild.
        if self.hot_start:
            self._hot_start()
            [sp.update_fixed_vars(self.fp) for sp in self.subproblems.values()]
            [sp.solve() for sp in self.subproblems.values()]
            self._add_cut()
            self._save_vars(self.fp)
            self.iter += 1
            self.hot_start = False

        def callback(m, where): #Define within the solve function to have access to the mp
            if where == GRB.Callback.MIPSOL:
                # 0. Update bounds
                ub = mp.data.upper_bounds[-1]
                lb = mp.data.lower_bounds[-1]
                ub_min = min(mp.data.upper_bounds)
                gap = (ub_min - lb)  / lb * 100
                print(f'>>> Iteration {mp.iter}. UB: {int(ub/1e6)} | LB: {int(lb/1e6)} | Gap: {gap} %')
                # 1. Save the variables of the current mp-solution
                mp._save_vars()
                # 2. Check termination criterias: Relative gap, Absolute gap & Number of iterations 
                    # 2.1 Relative gap < 1%
                if ub_min <= lb + 0.01*lb:
                    print(f'**OPTIMAL SOLUTION FOUND**')
                    m.terminate()
                    # 2.2 Absolute gap   
                elif ub_min - lb <= 1e6:
                    print(f'**ABSOLUTE GAP**')
                    print(mp.iter)
                    m.terminate()
                    # 2.3 Number of iterations
                elif mp.iter > mp.MAX_ITERS:
                    print(f'**MAX ITERS**')
                    print(mp.iter)
                    m.terminate()

                # 2. Solve subproblems
                [sp.update_fixed_vars() for sp in mp.subproblems.values()]
                [sp.solve() for sp in mp.subproblems.values()]

                # 3. Add a cut to the mp
                mp._add_cut()

                # 4. Update the bounds on the mp
                mp._update_bounds()

                # 5. Updater the number of iterations
                mp.iter += 1

            return

        m.optimize(callback)
        """NEXT STEPS - to ensure that Bender's algorithm don't stop when MP finde optimal solution
        if bounds not met: 
            resolve subproblems 
            add new cut
            m.optize(callback)
        """

        return
    
    def _hot_start(self):

        self.fp = Full_problem(self)
        self.fp.solve()

        return
            
    def _add_cut(self):
        m = self.m

        #Imports sets and other necessary data
        V = self.data.V
        P = self.data.P
        N = self.data.N
        S_n = self.data.S_n
        PROB_SCENARIO = self.data.PROB_SCENARIO

        # Define dictionaries for sensitivities and objective values of the subproblems
        z_sub = dict.fromkeys(N)
        prob_node = dict.fromkeys(N)
        sens_ports = pd.DataFrame(data = np.zeros(shape=(len(P),len(N))))
        sens_vessels = pd.DataFrame(data = np.zeros(shape=(len(V),len(N))))

        for n in N:
            #Calculate probability for being in a node by summing up scenarios where the node is present
            prob_node[n] = sum([PROB_SCENARIO.iloc[0,s] for s in S_n[n]])
            # Get the probability adjusted objective values of the subproblems 
            z_sub[n] = self.subproblems[n].m.objVal
            for v in V:
                sens_vessels.iloc[v,n] = self.subproblems[n].constraints.fix_vessels[(v,n)].pi
            for i in P:
                sens_ports.iloc[i,n] = self.subproblems[n].constraints.fix_ports[(i,n)].pi
        
        # Generate cut
        lhs = self.variables.phi
        rhs = gp.quicksum(prob_node[n]*(z_sub[n]+
        gp.quicksum(sens_vessels.iloc[v,n] * (self.variables.vessels[(v,n)]-self.subproblems[n].variables.vessels_free[(v,n)].x) for v in V) +
        gp.quicksum(sens_ports.iloc[i,n] * (self.variables.ports[(i,n)]-self.subproblems[n].variables.ports_free[(i,n)].x) for i in P[1:]))
        for n in N)

        if not self.hot_start: #If not being in the HOT START run, add the constraints as a lazy constraint
            m.cbLazy(lhs >= rhs)
        else:
            self.constraints.hot_start_cut = m.addConstr(lhs >= rhs)
        return

    ###
    # 
    ####
    def _update_bounds(self):
        m = self.m

        N = self.data.N
        S_n = self.data.S_n

        #Fetch the current value of the master problem and the artificial variable phi at the current MIPSOL in the callback
        z_master = m.cbGet(GRB.Callback.MIPSOL_OBJ)
        phi_val = m.cbGetSolution(self.variables.phi)

        PROB_SCENARIO = self.data.PROB_SCENARIO

        z_sub_total = 0

        for n in N:
            #Calculate probability for being in a node by summing up scenarios where the node is present
            prob_node = sum([PROB_SCENARIO.iloc[0,s] for s in S_n[n]])
            # Get the probability adjusted objective values of the subproblems 
            z_sub = self.subproblems[n].m.objVal
            z_sub_total += z_sub*prob_node

        # The best upper bound is the best incumbent with phi replaced by the sub problems' actual cost
        self.data.ub = z_master - phi_val + z_sub_total

        # The best lower bound is the current bestbound,
        # This will equal z_master at optimality
        try:
            self.data.lb = m.cbGet(GRB.Callback.MIPSOL_OBJBND)
        except gp.GurobiError:
            self.data.lb = z_master

        self.data.upper_bounds.append(self.data.ub)
        self.data.lower_bounds.append(self.data.lb)
        #self.data.mipgap.append(self.m.params.IntFeasTol)
        #self.data.solvetime.append(self.m.Runtime)

        return

    def _save_vars(self, model = None):
        if model == None:
            self.data.phis.append(self.m.cbGetSolution(self.variables.phi))

            for n in self.data.N:
                for v in self.data.V:
                    self.data.vessels[(v,n)].append(self.m.cbGetSolution(self.variables.vessels[(v,n)]))
                for i in self.data.P:
                    self.data.ports[(i,n)].append(self.m.cbGetSolution(self.variables.ports[(i,n)]))
        else: 
            for n_solved in self.data.N_s[model.data.S[0]]:
                N = get_same_year_nodes(n_solved, self.data.N, self.data.YEAR_OF_NODE)
                for n in N:
                    for v in self.data.V:
                        self.data.vessels[(v,n)].append(model.variables.vessels[(v,n_solved)].x)
                    for i in self.data.P:
                        self.data.ports[(i,n)].append(model.variables.ports[(i,n_solved)].x)

        return