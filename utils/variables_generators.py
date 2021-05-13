def make_routes_vessels_variables(V, R_v, T, S):
    constrs = []
    for v in V:
        for r in R_v[v]:
            for t in T:
                for s in S:
                    constrs.append((v, r, t, s))
    return constrs


def make_weekly_routes_vessels_variables(V, R_v, T, W, S):
    constrs = []
    for v in V:
        for r in R_v[v]:
            for t in T:
                for w in W:
                    for s in S:
                        constrs.append((v, r, t, w, s))
    return constrs


make_weekly_routes_vessels_variables


def make_delivery_vessel_variables(V, R_v, P_r, T, W, S):
    constrs = []
    for v in V:
        for r in R_v[v]:
            for i in P_r[r]:
                for t in T:
                    for w in W:
                        for s in S:
                            constrs.append((i, v, r, t, w, s))
    return constrs


def make_delivery_truck_variables(P, K_i, T, W, S):
    constrs = []
    for i in P:
        for k in K_i[i]:
            for t in T:
                for w in W:
                    for s in S:
                        constrs.append((i, k, t, w, s))
    return constrs
