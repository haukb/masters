import numpy as np
import dill

def get_same_year_nodes(node, N, YEAR_OF_NODE):
    year = YEAR_OF_NODE.iloc[0,node]
    nodes = []
    for n in N:
        if year == YEAR_OF_NODE.iloc[0,n]:
            nodes.append(n)

    return nodes

def nodes_with_new_investments(mp2sp_iterations, mp_iter, vessels, ports, V, P, N, NP_n):
    N_changed = list(N.copy())
    for n in N:
        cum_vessels_latest = np.array([sum([vessels[(v,m)][-1] for m in NP_n[n]]) for v in V])
        cum_ports_latest = np.array([sum([ports[(i,m)][-1] for m in NP_n[n]]) for i in P])
        mp2sp_iterations[mp_iter,n] = max(mp2sp_iterations[:,n])+1 #Default option, given that node solution is new

        for j in range(mp_iter): #The number of MP iterations
            cum_vessels = np.array([sum([vessels[(v,m)][j] for m in NP_n[n]]) for v in V])
            cum_ports = np.array([sum([ports[(i,m)][j] for m in NP_n[n]]) for i in P])
            if (cum_vessels_latest==cum_vessels).all() and (cum_ports_latest==cum_ports).all():
                N_changed.remove(n)
                mp2sp_iterations[mp_iter,n] = mp2sp_iterations[j,n]
                break

    return N_changed, mp2sp_iterations

def get_obj(run_id):
    with open(f'Results/{run_id}.obj', 'rb') as fobj:
        # do somdthing with fobj
        data = dill.load(fobj)
    return data