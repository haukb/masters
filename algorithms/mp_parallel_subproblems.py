import ray
from time import time

from utils.master_problem_template import Master_problem, expando
from utils.full_problem import Full_problem
from subproblems.subproblem_parallel import Subproblem_parallel

from utils.misc_functions import nodes_with_new_investments


class MP_parallelSPs(Master_problem):
    def __init__(self, kwargs) -> None:
        super().__init__(**kwargs)

    def _make_subproblems(self) -> None:
        self.sp_refs = {
            n: Subproblem_parallel.remote(n, self.data) for n in self.data.N
        }
        self.subproblems = {n: expando() for n in self.data.N}
        return

    def solve(self) -> None:
        run_start = time()
        ray.init()
        m = self.m
        # mp2sp_iterations = np.zeros([self.MAX_ITERS,self.data.NUM_NODES], dtype=int)

        # Build the subproblems as remote actors in ray,
        # and store the references in the master problem
        self._make_subproblems()

        # Warm start algorithm with solution from deteministic problem
        if self.warm_start:
            self._warm_start()

        while True:
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
            # N_changed, mp2sp_iterations = nodes_with_new_investments(mp2sp_iterations,
            # self.iter, self.data.vessels, self.data.ports, self.data.V, self.data.P,
            # self.data.N, self.data.NP_n)
            # N_4bounds = mp2sp_iterations[self.iter,:]
            N_changed = self.data.N
            N_4bounds = [self.iter for n in self.data.N]

            [self.sp_refs[n]._update_fixed_vars.remote(self.data) for n in N_changed]

            updated_sps = ray.get([self.sp_refs[n].solve.remote() for n in N_changed])
            for idx, updated_sp in enumerate(updated_sps):
                self.subproblems[N_changed[idx]].data = updated_sp

            t1 = time()
            self.data.sp_solve_time.append(t1 - t0)
            print(f"Time spent solving SP: {round(t1-t0,3)}")

            # 3. Update the bounds on the mp
            self._update_bounds(N_4bounds)

            # 4. Check termination criterias:
            # Relative gap, Absolute gap & Number of iterations
            if self._check_termination(run_start):
                break
            else:
                # 5. Add a cut to the mp and update the allowed vessel changes
                self._add_cut(N_changed)
                # self._update_vessel_changes()
                self.iter += 1

        ray.shutdown()

        # This line also save the sp data, directly to the mp' data, so that it can be access in later analysis
        for n in self.data.N:
            self.data.sp[n] = self.subproblems[n].data
        return

    def _warm_start(self) -> None:
        t0 = time()

        self.fp = Full_problem(self, SCENARIOS=[4])
        self.fp.solve()
        self._save_vars(self.fp)

        [self.sp_refs[n]._update_fixed_vars.remote(self.data) for n in self.data.N]
        sp_data = ray.get([self.sp_refs[n].solve.remote() for n in self.data.N])
        for n in self.data.N:
            self.subproblems[n].data = sp_data[n]

        self._add_cut(self.data.N)
        # self._update_vessel_changes(self.fp)
        self.warm_start = False

        t1 = time()
        self.data.warm_start_solve_time = t1 - t0
        self.iter += 1
        return
