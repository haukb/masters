from utils.master_problem_template import Master_problem
from utils.economic_analysis import run_economic_analysis
from utils.misc_functions import nodes_with_new_investments

from time import time


class MP_2opt(Master_problem):
    def __init__(self, kwargs) -> None:
        super().__init__(**kwargs)

    def solve(self):
        run_start = time()
        m = self.m
        # mp2sp_iterations = np.zeros([self.MAX_ITERS,self.data.NUM_NODES], dtype=int)

        self._make_subproblems()

        # Warm start algorithm with solution from deteministic problem
        if self.warm_start:
            self._warm_start()

        while True:
            self.iter += 1
            # 1. Solve master problem and save variables
            t0 = time()
            m.optimize()
            t1 = time()

            self._save_vars()
            self.data.mp_solve_time.append(t1 - t0)
            print(f"\n>>>ITERATION {self.iter}")
            print(f"Time spent solving MP: {round(t1-t0,3)}")

            # 2. Solve subproblems
            t0 = time()
            # N_changed, mp2sp_iterations = nodes_with_new_investments(mp2sp_iterations, mp.iter, mp.data.vessels, mp.data.ports, mp.data.V, mp.data.P, mp.data.N, mp.data.NP_n)

            for n in self.data.N:  # OLD replace N_changed with mp.data.N
                self.subproblems[n]._update_fixed_vars()
                self.subproblems[n].solve()
            t1 = time()
            self.data.sp_solve_time.append(t1 - t0)
            print(f"Time spent solving SP: {round(t1-t0,3)}")

            # print(f'Obj. vals. from similar SP: {[int(self.subproblems[n].m.ObjVal*1e-6) for n in self.data.N]}')

            # 3. Update the bounds on the mp
            self._update_bounds()

            # 4. Check termination criterias
            if self._check_termination(run_start):
                break

            # 5. Add a cut to the mp and update the allowed vessel changes
            self._add_cut(self.data.N)
            # self._update_vessel_changes()

        for n in self.data.N:
            self.sp_data[n] = self.subproblems[n].data
        self.run_economic_analysis()
        return
