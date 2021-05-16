import ray
from subproblems.subproblem import Subproblem


# Subproblem
@ray.remote
class Subproblem_parallel(Subproblem):
    def __init__(self, NODE, data) -> None:
        super().__init__(NODE=NODE, data=data, PARALLEL=True)
        return

    def _update_fixed_vars(self, updated_data):

        for n in self.mp.data.N:
            for i in self.mp.data.P:
                port_val = updated_data.ports[(i, n)][-1]
                self.constraints.fix_ports[(i, n)].rhs = port_val
            for v in self.mp.data.V:
                vessel_val = updated_data.vessels[(v, n)][-1]
                self.constraints.fix_vessels[(v, n)].rhs = vessel_val

        return
