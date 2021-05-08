from math import isnan
from numpy.lib.arraysetops import unique
import pandas as pd
import numpy as np

def arc_set_generator(ROUTES, R):
    #Preallocate list of lists
    A = [[] for r in R]
    
    for r in R:
        route = iter(ROUTES.iloc[:,r].tolist())
        prev_port = next(route)
        port = next(route)
        while not isnan(port):
            A[r].append((prev_port,port))
            prev_port = port
            if prev_port == 0:
                break
            try:
                port = next(route)
            except:
                break
    return A

def port_route_set_generator(ROUTES,P,R):
    #Preallocate list of lists
    P_r = [[] for r in R]
    
    for r in R:
        route = iter(ROUTES.iloc[:,r].tolist())
        port = next(route)
        P_r[r].append(port)
        port = next(route)
        while not isnan(port) and port != 0:
            P_r[r].append(port)
            try:
                port = next(route)
            except:
                break

    #not_P_r = [[port for port in P if port not in P_r[route] and port != 0] for route in R]

    return P_r#, not_P_r

def route_vessel_set_generator(ROUTE_FEASIBILITY, V):
    R_v = [[] for v in V]
    not_R_v = [[] for v in V]
    for v in V:
        binary_list = ROUTE_FEASIBILITY.iloc[v,:].tolist()
        for i,route in enumerate(binary_list):
            if route != 0:
                R_v[v].append(i)
            #else:
            #    not_R_v[v].append(i)   

    return R_v#, not_R_v 

def route_vessel_port_set_generator(ROUTES, R_v, V, P):
    R_vi = np.empty((len(V), len(P)), dtype=object)
    for i in np.ndindex(R_vi.shape): 
        R_vi[i] = []

    for v in V:
        routes_by_vessel = R_v[v]
        ports_list = [ROUTES.iloc[:,r].tolist() for r in routes_by_vessel] # [1,0]
        for i in P:
            for ports,route in zip(ports_list, routes_by_vessel):
                if i in ports: 
                    R_vi[v,i].append(route)
    return R_vi

def port_customer_set_generator(PORT_CUSTOMER_FEASIBILITY, P, K):
    P_k = [[] for k in K]
    for p in P: 
        customers = [int(x) for x in PORT_CUSTOMER_FEASIBILITY.iloc[:,p].dropna()]
        for k in customers: 
            P_k[k].append(p)
    
    return P_k

def beta_set_generator(YEAR_OF_NODE, NUM_NODES, NUM_YEARS):
    BETA = pd.DataFrame(np.zeros([NUM_NODES, NUM_YEARS]))
    for node in range(NUM_NODES):
        idx = YEAR_OF_NODE.iloc[0,node]
        BETA.iloc[node, idx:] = 1
    return BETA

def scenario_node_set_generator(NODES_IN_SCENARIO, N, S):
    S_n = [[] for n in N]
    for s in S:
        for n in N: 
            if n in NODES_IN_SCENARIO.iloc[:,s].values:
                S_n[n].append(s)
    return S_n

def year_node_set_generator(NODES_IN_SCENARIO, YEAR_OF_NODE, NUM_YEARS, N):
    investment_node_years = np.append(NODES_IN_SCENARIO.index.to_numpy(), NUM_YEARS)
    year_span_dict = {}
    for (a,b) in zip(investment_node_years[:-1], investment_node_years[1:]):
        year_span_dict[a] = np.arange(a,b)
    T_n = [year_span_dict[YEAR_OF_NODE.iloc[0,n]] for n in N]
    
    return T_n

def parent_node_set_generator(NODES_IN_SCENARIO, N):
    NP_n = [[] for n in N]

    for i in range(NODES_IN_SCENARIO.shape[0]):
        nodes_in_year = NODES_IN_SCENARIO.iloc[i,:].tolist()
        for n in set(nodes_in_year):
            idx = nodes_in_year.index(n)
            NP_n[n] = NODES_IN_SCENARIO.iloc[:i+1, idx].tolist()
         
    return NP_n