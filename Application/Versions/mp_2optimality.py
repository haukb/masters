from master_problem import Master_problem
from subproblem import Subproblem
from misc_functions import nodes_with_new_investments

from time import time
import numpy as np

class MP_2opt(Master_problem):
    def __init__(self, INSTANCE, NUM_WEEKS = 1, NUM_SCENARIOS = 27, NUM_VESSELS = 2, MAX_PORT_VISITS = 2, DRAW = False, WEEKLY_ROUTING = False, DISCOUNT_FACTOR = 1, BENDERS_GAP=0.01, MAX_ITERS=100, warm_start = True) -> None:
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
    
    def solve(self):
        mp = self
        m = self.m
        mp2sp_iterations = np.zeros([self.MAX_ITERS,self.data.NUM_NODES], dtype=int)

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
            self.data.mp_solve_time.append(t1-t0)
            print(f'\n>>>ITERATION {mp.iter}')
            print(f'>>>Time spent solving MP: {round(t1-t0,3)}')

            # 2. Solve subproblems
            t0 = time()
            N_changed, mp2sp_iterations = nodes_with_new_investments(mp2sp_iterations, mp.iter, mp.data.vessels, mp.data.ports, mp.data.V, mp.data.P, mp.data.N, mp.data.NP_n)

            for n in mp.data.N: #OLD replace N_changed with mp.data.N 
                mp.subproblems[n].update_fixed_vars()
                mp.subproblems[n].solve()
            t1 = time()
            self.data.sp_solve_time.append(t1-t0)
            print(f'Time spent solving SP: {round(t1-t0,3)}')

            #print(f'Obj. vals. from similar SP: {[int(self.subproblems[n].m.ObjVal*1e-6) for n in self.data.N]}')

                # 3. Update the bounds on the mp
            mp._update_bounds()

            # 4. Check termination criterias: Relative gap, Absolute gap & Number of iterations 
            try: 
                lb = mp.data.lower_bounds[-1] #lb increasing in each iteration, lb max is the last element
                ub = min(mp.data.upper_bounds) #ub is the lowest ub found up until the current iteration.
                gap = (ub - lb)  / lb * 100
                print(f'BOUNDS: UB = {int(ub/1e6)} | LB = {int(lb/1e6)} | Gap = {round(gap,2)} %')
            except:
                print(f'Iteration {mp.iter}. Bounds not applicable')

                # 4.1 Relative gap < 1%
            if ub <= lb + 0.0001*lb:
                print(f'**OPTIMAL SOLUTION FOUND: {int(ub*1e-6)}**')
                break
                # 4.2 Absolute gap   
            elif ub - lb <= 1e6:
                print(f'**ABSOLUTE GAP**')
                print(mp.iter)
                break
                # 4.3 Number of iterations
            elif mp.iter > mp.MAX_ITERS:
                print(f'**MAX ITERATIONS REACHED {mp.MAX_ITERS}**')
                break

            # 5. Add a cut to the mp and update the allowed vessel changes
            #mp._add_unicut()
            mp._add_multicut(mp.data.N)
            mp._update_vessel_changes()
        
        self.data.sp_zVals = np.ones([mp.iter, self.data.NUM_NODES])
        for i in range(mp.iter):
            for n in self.data.N:
                self.data.sp_zVals[i,n] = int(self.subproblems[n].data.obj_vals[mp2sp_iterations[i,n]]*1e-6)

        return
