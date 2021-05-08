from utils.master_problem_template import Master_problem
from subproblems.subproblem import Subproblem
from utils.misc_functions import nodes_with_new_investments

from time import time
import numpy as np

class MP_2opt(Master_problem):
    def __init__(self, INSTANCE, NUM_WEEKS = 1, NUM_SCENARIOS = 27, NUM_VESSELS = 2, MAX_PORT_VISITS = 2, DRAW = False, WEEKLY_ROUTING = False, DISCOUNT_FACTOR = 1, BENDERS_GAP=0.01, MAX_ITERS=100, TIME_LIMIT=7600, warm_start = True) -> None:
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
        TIME_LIMIT=TIME_LIMIT, 
        warm_start=warm_start)
    
    def solve(self):
        run_start = time()
        m = self.m
        #mp2sp_iterations = np.zeros([self.MAX_ITERS,self.data.NUM_NODES], dtype=int)

        # Only build subproblems if they don't exist or a rebuild is forced.
        if not hasattr(self, 'subproblems'):# or force_submodel_rebuild:
            self.subproblems = {n: Subproblem(NODE=n, mp=self) for n in self.data.N}
        
        #Warm start algorithm with solution from deteministic problem
        if self.warm_start:
            self._warm_start()

        while True: 
            self.iter += 1
            # 1. Solve master problem and save variables
            t0 = time()
            m.optimize()
            t1 = time()

            self._save_vars()
            self.data.mp_solve_time.append(t1-t0)
            print(f'\n>>>ITERATION {self.iter}')
            print(f'Time spent solving MP: {round(t1-t0,3)}')

            # 2. Solve subproblems
            t0 = time()
            #N_changed, mp2sp_iterations = nodes_with_new_investments(mp2sp_iterations, mp.iter, mp.data.vessels, mp.data.ports, mp.data.V, mp.data.P, mp.data.N, mp.data.NP_n)

            for n in self.data.N: #OLD replace N_changed with mp.data.N 
                self.subproblems[n].update_fixed_vars()
                self.subproblems[n].solve()
            t1 = time()
            self.data.sp_solve_time.append(t1-t0)
            print(f'Time spent solving SP: {round(t1-t0,3)}')

            #print(f'Obj. vals. from similar SP: {[int(self.subproblems[n].m.ObjVal*1e-6) for n in self.data.N]}')

                # 3. Update the bounds on the mp
            self._update_bounds()

            # 4. Check termination criterias
            if self._check_termination(run_start):
                break

            # 5. Add a cut to the mp and update the allowed vessel changes
            #mp._add_unicut()
            self._add_multicut(self.data.N)
            #mp._update_vessel_changes()

        return
