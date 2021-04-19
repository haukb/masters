import pandas as pd
import numpy as np

def get_same_year_nodes(node, N, YEAR_OF_NODE):
    year = YEAR_OF_NODE.iloc[0,node]
    nodes = []
    for n in N:
        if year == YEAR_OF_NODE.iloc[0,n]:
            nodes.append(n)

    return nodes

def nodes_with_new_investments(vessels, ports, V, P, N, NP_n):
    new_nodes = list(N.copy())
    iterations = range(len(ports[(0,0)])-1)
    for n in N:
        cum_vessels_latest = np.array([sum([vessels[(v,m)][-1] for m in NP_n[n]]) for v in V])
        cum_ports_latest = np.array([sum([ports[(i,m)][-1] for m in NP_n[n]]) for i in P])
        for j in iterations:
            cum_vessels = np.array([sum([vessels[(v,m)][j] for m in NP_n[n]]) for v in V])
            cum_ports = np.array([sum([ports[(i,m)][j] for m in NP_n[n]]) for i in P])
            if (cum_vessels_latest==cum_vessels).all() and (cum_ports_latest==cum_ports).all():
                new_nodes.remove(n)
                break   

    return new_nodes