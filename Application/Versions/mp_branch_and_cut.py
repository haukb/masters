from master_problem import Master_problem
from subproblem import Subproblem

from misc_functions import get_same_year_nodes, nodes_with_new_investments

import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

class MP_BBC(Master_problem):
    def __init__(self, INSTANCE, NUM_WEEKS = 1, NUM_SCENARIOS = 27, NUM_VESSELS = 1, MAX_PORT_VISITS = 3, DRAW = False, WEEKLY_ROUTING = False, DISCOUNT_FACTOR = 1, BENDERS_GAP=0.01, MAX_ITERS=10, warm_start = True) -> None:
        super().__init__(INSTANCE, 
        NUM_WEEKS=NUM_WEEKS, 
        NUM_SCENARIOS=NUM_SCENARIOS, 
        NUM_VESSELS=NUM_VESSELS, 
        MAX_PORT_VISITS=MAX_PORT_VISITS, 
        DRAW=DRAW, 
        WEEKLY_ROUTING=WEEKLY_ROUTING, 
        DISCOUNT_FACTOR=DISCOUNT_FACTOR, 
        BENDERS_GAP=BENDERS_GAP, 
        MAX_ITERS=MAX_ITERS, 
        warm_start=warm_start)

    def _build_constraints(self):
        #Only open port in one time, to hinder "double" opening for dual values in several nodes
        self.constraints.max_ports_in_scenario = self.m.addConstrs(gp.quicksum(self.variables.ports[(i,n)] for n in self.data.N_s[s]) <= 1 for i in self.data.P for s in self.data.S)
        self.constraints.max_number_of_vessels = self.m.addConstrs(gp.quicksum(self.variables.vessels[(v,n)] for n in self.data.N_s[s]) <= 10 for v in self.data.V for s in self.data.S)
        return 

    def solve(self) -> None:

        # Only build subproblems if they don't exist or a rebuild is forced.
        if not hasattr(self, 'subproblems'):# or force_submodel_rebuild:
            self.subproblems = {n: Subproblem(self, NODE=n) for n in self.data.N}

        # Update fixed variables for subproblems and rebuild.
        #if self.warm_start:
        #    self._warm_start()

        def callback(m, where): #Define within the solve function to have access to the mp
            # 1. Find the next IP-solution to the MP
            if where == GRB.Callback.MIPSOL:
                self.iter += 1
                # 2. Save the investment variables
                self._save_vars()
                # 3. Solve subproblems with the new investment variables 
                for n in self.data.N: #OLD replace N_changed with mp.data.N 
                    self.subproblems[n].update_fixed_vars_callback()
                    self.subproblems[n].solve()
                # 4. Update bounds
                self._update_bounds()
                # 5. Check termination
            
                lb = self.data.lower_bounds[-1] #lb increasing in each iteration, lb max is the last element
                ub = min(self.data.upper_bounds) #ub is the lowest ub found up until the current iteration.
                print(f'lb: {lb}, ub: {ub}')
                try:
                    gap = (ub - lb)  / lb * 100
                    print(f'> Iteration {self.iter}. UB: {int(ub/1e6)} | LB: {int(lb/1e6)} | Gap: {round(gap,2)} %')
                except:
                    print(f'> Iteration {self.iter}. Bounds not applicable')
                    
                # 5. if phi < second stage costs add new cut
                if ub >= lb + 0.001*lb:
                    self._add_multicut(self.data.N)
            return

        self.m.optimize(callback)

        return
    
    def _add_multicut(self, N):
        m = self.m

        #Imports sets and other necessary data
        V = self.data.V
        P = self.data.P
        NP_n = self.data.NP_n

        for n in N: 
            lhs = self.variables.phi[n]
            z_sub = self.subproblems[n].m.ObjVal
            sens_vessels = [self.subproblems[n].constraints.fix_vessels[(v,n)].pi for v in V]
            sens_ports = [self.subproblems[n].constraints.fix_ports[(i,n)].pi for i in P]
            rhs = (z_sub + 
            gp.quicksum(sens_vessels[v] * gp.quicksum(self.variables.vessels[(v,m)]-self.subproblems[n].variables.vessels_free[(v,m)].x for m in NP_n[n]) for v in V) + 
            gp.quicksum(sens_ports[i] * gp.quicksum(self.variables.ports[(i,m)]-self.subproblems[n].variables.ports_free[(i,m)].x for m in NP_n[n]) for i in P[1:]))
            m.cbLazy(lhs >= rhs)

        return

    
    def _update_bounds(self):
        m = self.m

        N = self.data.N

        #Fetch the current value of the master problem and the artificial variable phi at the current MIPSOL in the callback
        z_master = m.cbGet(GRB.Callback.MIPSOL_OBJ)
        phi_val = sum([self.m.cbGetSolution(self.variables.phi[n]) for n in N])
        z_sub_total = sum([self.subproblems[n].m.ObjVal for n in N])

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
                self.data.phis.append(self.m.cbGetSolution(self.variables.phi[n]))
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