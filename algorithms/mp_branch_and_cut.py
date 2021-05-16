import gurobipy as gp
from gurobipy import GRB
from time import time

from utils.master_problem_template import Master_problem
from subproblems.subproblem import Subproblem
from utils.misc_functions import get_same_year_nodes


class MP_BBC(Master_problem):
    def __init__(self, kwargs) -> None:
        super().__init__(**kwargs)

    def _build_constraints(self):
        # Only open port in one time, to hinder "double" opening for dual values in several nodes
        self.constraints.max_ports_in_scenario = self.m.addConstrs(
            gp.quicksum(self.variables.ports[(i, n)] for n in self.data.N_s[s]) <= 1
            for i in self.data.P
            for s in self.data.S
        )
        return

    def solve(self) -> None:
        run_start = time()
        # Only build subproblems if they don't exist or a rebuild is forced.
        if not hasattr(self, "subproblems"):  # or force_submodel_rebuild:
            self.subproblems = {n: Subproblem(NODE=n, mp=self) for n in self.data.N}

        # Update fixed variables for subproblems and rebuild.
        if self.warm_start:
            self._warm_start()

        def callback(
            m, where
        ):  # Define within the solve function to have access to the mp
            # 1. Find the next IP-solution to the MP
            if where == GRB.Callback.MIPSOL:
                self.data.mp_t1 = time()
                self.data.mp_solve_time.append(self.data.mp_t1 - self.data.mp_t0)
                self.iter += 1
                # 2. Save the investment variables
                self._save_vars()
                self.data.sp_t0 = time()
                # 3. Solve subproblems with the new investment variables
                for n in self.data.N:  # OLD replace N_changed with mp.data.N
                    self.subproblems[n]._update_fixed_vars_callback()
                    self.subproblems[n].solve()
                self.data.sp_t1 = time()
                self.data.sp_solve_time.append(self.data.sp_t1 - self.data.sp_t0)
                # 4. Update bounds
                self._update_bounds()
                # 5. Check termination
                if self._check_termination(run_start):
                    pass
                else:
                    self._add_lazy_cut(self.data.N)

                self.data.mp_t0 = time()

            return

        self.data.mp_t0 = time()
        self.m.optimize(callback)

        return

    def _add_lazy_cut(self, N):
        m = self.m

        # Imports sets and other necessary data
        V = self.data.V
        P = self.data.P
        NP_n = self.data.NP_n

        for n in N:
            lhs = self.variables.phi[n]
            z_sub = self.subproblems[n].m.ObjVal
            sens_vessels = [
                self.subproblems[n].constraints.fix_vessels[(v, n)].pi for v in V
            ]
            sens_ports = [
                self.subproblems[n].constraints.fix_ports[(i, n)].pi for i in P
            ]
            rhs = (
                z_sub
                + gp.quicksum(
                    sens_vessels[v]
                    * gp.quicksum(
                        self.variables.vessels[(v, m)]
                        - self.subproblems[n].variables.vessels_free[(v, m)].x
                        for m in NP_n[n]
                    )
                    for v in V
                )
                + gp.quicksum(
                    sens_ports[i]
                    * gp.quicksum(
                        self.variables.ports[(i, m)]
                        - self.subproblems[n].variables.ports_free[(i, m)].x
                        for m in NP_n[n]
                    )
                    for i in P[1:]
                )
            )
            m.cbLazy(lhs >= rhs)

        return

    def _update_bounds(self):
        m = self.m

        N = self.data.N

        # Fetch the current value of the master problem and the artificial variable phi at the current MIPSOL in the callback
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

    def _save_vars(self, model=None):
        if model == None:

            for n in self.data.N:
                self.data.phis.append(self.m.cbGetSolution(self.variables.phi[n]))
                for v in self.data.V:
                    self.data.vessels[(v, n)].append(
                        self.m.cbGetSolution(self.variables.vessels[(v, n)])
                    )
                for i in self.data.P:
                    self.data.ports[(i, n)].append(
                        self.m.cbGetSolution(self.variables.ports[(i, n)])
                    )
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
