import gurobipy as gp
import pandas as pd
import numpy as np

from algorithms.mp_2optimality import MP_2opt


class MP_unicut(MP_2opt):
    def __init__(self, kwargs) -> None:
        super().__init__(kwargs)

    def _add_cut(self, N):
        m = self.m

        # Imports sets and other necessary data
        V = self.data.V
        P = self.data.P
        NP_n = self.data.NP_n

        # Define dictionaries for sensitivities and objective values of the subproblems
        z_sub = dict.fromkeys(N)
        sens_ports = pd.DataFrame(data=np.zeros(shape=(len(P), len(N))))
        sens_vessels = pd.DataFrame(data=np.zeros(shape=(len(V), len(N))))

        for n in N:
            # Get the probability adjusted objective values of the subproblems
            z_sub[n] = self.subproblems[n].m.ObjVal
            for v in V:
                sens_vessels.iloc[v, n] = (
                    self.subproblems[n].constraints.fix_vessels[(v, n)].pi
                )
            for i in P:
                sens_ports.iloc[i, n] = (
                    self.subproblems[n].constraints.fix_ports[(i, n)].pi
                )

        # Generate cut
        lhs = self.variables.phi[0]
        rhs = gp.quicksum(
            z_sub[n]
            + gp.quicksum(
                sens_vessels.iloc[v, n]
                * gp.quicksum(
                    self.variables.vessels[(v, m)]
                    - self.subproblems[m].variables.vessels_free[(v, m)].x
                    for m in NP_n[n]
                )
                for v in V
            )
            + gp.quicksum(
                sens_ports.iloc[i, n]
                * (
                    self.variables.ports[(i, n)]
                    - self.subproblems[n].variables.ports_free[(i, n)].x
                )
                for i in P[1:]
            )
            for n in N
        )

        m.addConstr(lhs >= rhs)

        return
