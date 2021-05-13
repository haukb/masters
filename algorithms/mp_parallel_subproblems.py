import gurobipy as gp
from gurobipy import GRB
import ray
from time import time
import numpy as np

from utils.master_problem_template import Master_problem
from utils.full_problem import Full_problem
from subproblems.subproblem_parallel import Subproblem_parallel

from utils.misc_functions import nodes_with_new_investments


class MP_parallelSPs(Master_problem):
    def __init__(self, kwargs) -> None:
        super().__init__(**kwargs)

    def _make_subproblems(self):
        self.sp_refs = {
            n: Subproblem_parallel.remote(n, self.data) for n in self.data.N
        }
        return

    def solve(self) -> None:
        run_start = time()
        ray.init()
        m = self.m
        self.sp_data = dict.fromkeys(list(self.data.N))
        # mp2sp_iterations = np.zeros([self.MAX_ITERS,self.data.NUM_NODES], dtype=int)

        # Build the subproblems as remote actors in ray, and store the references in the master problem
        self._make_subproblems()

        # Warm start algorithm with solution from deteministic problem
        if self.warm_start:
            self._warm_start()

        sps_solved = 13

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
            # N_changed, mp2sp_iterations = nodes_with_new_investments(mp2sp_iterations, self.iter, self.data.vessels, self.data.ports, self.data.V, self.data.P, self.data.N, self.data.NP_n)
            sps_solved += 13  # len(N_changed)
            # N_4bounds = mp2sp_iterations[self.iter,:]
            N_changed = self.data.N
            N_4bounds = [self.iter for n in self.data.N]

            [self.sp_refs[n].update_fixed_vars.remote(self.data) for n in N_changed]

            updated_sps = ray.get([self.sp_refs[n].solve.remote() for n in N_changed])
            for idx, updated_sp in enumerate(updated_sps):
                self.sp_data[N_changed[idx]] = updated_sp

            t1 = time()
            self.data.sp_solve_time.append(t1 - t0)
            print(f"Time spent solving SP: {round(t1-t0,3)}")

            # 3. Update the bounds on the mp
            self._update_bounds(self.sp_data, N_4bounds)

            # 4. Check termination criterias: Relative gap, Absolute gap & Number of iterations
            if self._check_termination(run_start):
                break
            else:
                # 5. Add a cut to the mp and update the allowed vessel changes
                self._add_cut(N_changed, self.sp_data)
                # self._update_vessel_changes()
                self.iter += 1

        ray.shutdown()
        print(f"Solved {sps_solved} subproblems")
        return

    def _warm_start(self):
        t0 = time()

        self.fp = Full_problem(self, SCENARIOS=[4])
        self.fp.solve()
        self._save_vars(self.fp)

        [self.sp_refs[n].update_fixed_vars.remote(self.data) for n in self.data.N]
        sp_data = ray.get([self.sp_refs[n].solve.remote() for n in self.data.N])
        for n in self.data.N:
            self.sp_data[n] = sp_data[n]

        self._add_cut(self.data.N, self.sp_data)
        # self._update_vessel_changes(self.fp)
        self.warm_start = False

        t1 = time()
        self.data.warm_start_solve_time = t1 - t0
        self.iter += 1
        return

    def _add_cut(self, N, sp_data):
        m = self.m

        # Imports sets and other necessary data
        V = self.data.V
        P = self.data.P
        NP_n = self.data.NP_n

        for n in N:
            lhs = self.variables.phi[n]
            z_sub = sp_data[n].obj_vals[-1]
            sens_vessels = sp_data[n].sens_vessels[-1]
            sens_ports = sp_data[n].sens_ports[-1]
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

    def _update_bounds(self, sp_data, N_4bounds):
        m = self.m
        N = self.data.N

        # Fetch the current value of the master problem and the artificial variable phi at the current MIPSOL in the callback
        z_master = m.ObjVal
        phi_val = sum(
            [self.variables.phi[n].x for n in N]
        )  # Get the latest phi as this is solved by the MP with new cuts
        z_sub_total = sum(
            [sp_data[n].obj_vals[N_4bounds[n]] for n in N]
        )  # Get actual cost of second stage as unchanged SPs are not resolved. N_4bounds =-1 if new iteration of SP

        # The best upper bound is the best incumbent with phi replaced by the sub problems' actual cost
        ub = z_master - phi_val + z_sub_total

        # The best lower bound is the current bestbound,
        # This will equal z_master at optimality
        lb = z_master

        self.data.upper_bounds.append(ub)
        self.data.lower_bounds.append(lb)

        return
