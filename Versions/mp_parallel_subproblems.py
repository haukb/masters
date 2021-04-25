import gurobipy as gp
from gurobipy import GRB
import ray
from time import time

from master_problem import Master_problem
from full_problem import Full_problem
from subproblem_parallel import Subproblem_parallel

from misc_functions import nodes_with_new_investments


class MP_parallelSPs(Master_problem):
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

    def solve(self) -> None:
        ray.init()
        mp = self
        m = self.m
        self.sp_data = [[] for n in self.data.N]
        #mp_data_id = ray.put(mp.data)
        #mp_variables_id = ray.put(mp.variables)

        #Build the subproblems as remote actors in ray, and store the references in the master problem
        self.sp_refs = {n: Subproblem_parallel.remote(n, mp.data) for n in self.data.N}
        
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
            [self.sp_refs[n].update_fixed_vars.remote(mp.data) for n in N_changed]

            updated_sps = ray.get([self.sp_refs[n].solve.remote() for n in N_changed])
            for idx, updated_sp in enumerate(updated_sps):
                self.sp_data[N_changed[idx]] = updated_sp

            t1 = time()
            print(f'>>Time spent solving SP: {round(t1-t0,3)}')

                # 3. Update the bounds on the mp
            mp._update_bounds(self.sp_data)

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
            mp._add_multicut(N_changed, self.sp_data)
            mp._update_vessel_changes()

        ray.shutdown()   
        return

    def _warm_start(self):

        self.fp = Full_problem(self, SCENARIOS = [4])
        self.fp.solve()
        self._save_vars(self.fp)

        [self.sp_refs[n].update_fixed_vars.remote(self.data) for n in self.data.N]
        self.sp_data = ray.get([self.sp_refs[n].solve.remote() for n in self.data.N])
        #self._add_unicut()
        self._add_multicut(self.data.N, self.sp_data)
        self._update_vessel_changes(self.fp)
        self.warm_start = False

        return
    
    def _add_multicut(self, N, sp_data):
        m = self.m

        #Imports sets and other necessary data
        V = self.data.V
        P = self.data.P
        NP_n = self.data.NP_n

        for n in N: 
            lhs = self.variables.phi[n]
            z_sub = sp_data[n].obj_vals[-1]
            sens_vessels = sp_data[n].sens_vessels[-1]
            sens_ports = sp_data[n].sens_ports[-1]
            rhs = (z_sub + 
            gp.quicksum(sens_vessels[v] * gp.quicksum(self.variables.vessels[(v,m)]-self.data.vessels[(v,m)][-1] for m in NP_n[n]) for v in V) + 
            gp.quicksum(sens_ports[i] * gp.quicksum(self.variables.ports[(i,m)]-self.data.ports[(i,m)][-1] for m in NP_n[n]) for i in P[1:]))
            m.addConstr(lhs >= rhs)

        return

    def _update_bounds(self, sp_data):
        m = self.m

        N = self.data.N

        #Fetch the current value of the master problem and the artificial variable phi at the current MIPSOL in the callback
        z_master = m.ObjVal
        phi_val = sum([self.variables.phi[n].x for n in N])
        z_sub_total = sum([sp_data[n].obj_vals[-1] for n in N])

        # The best upper bound is the best incumbent with phi replaced by the sub problems' actual cost
        self.data.ub = z_master - phi_val + z_sub_total

        # The best lower bound is the current bestbound,
        # This will equal z_master at optimality
        self.data.lb = z_master

        self.data.upper_bounds.append(self.data.ub)
        self.data.lower_bounds.append(self.data.lb)

        return
