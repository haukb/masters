# Standard libraries

# Special libraries
import gurobipy as gp

# Local imports


def _find_best_iteration(model):
    ub_min = min(model.data.upper_bounds)
    opt_sol_idx = model.data.upper_bounds.index(ub_min)
    return opt_sol_idx


def _calculate_vessel_opex(model, opt_sol_idx):

    N = model.data.N
    P_r = model.data.P_r
    V = model.data.V
    W = model.data.W
    T_n = model.data.T_n
    S_n = model.data.S_n
    R_v = model.data.R_v

    NUM_WEEKS = model.data.NUM_WEEKS
    PROB_SCENARIO = model.data.PROB_SCENARIO
    SAILING_COST = model.data.SAILING_COST
    PORT_HANDLING = model.data.PORT_HANDLING

    sailing_cost = 0
    container_handling_cost = 0
    for n in N:
        routes_vessels = model.sp_data[n].routes_vessels[opt_sol_idx]
        delivery_vessel = model.sp_data[n].delivery_vessel[opt_sol_idx]
        pickup_vessel = model.sp_data[n].pickup_vessel[opt_sol_idx]

        T_star = T_n[n]
        S_star = S_n[n]

        sailing_cost += (52 / NUM_WEEKS) * gp.quicksum(
            PROB_SCENARIO.iloc[0, s]
            * gp.quicksum(
                SAILING_COST[v, r, t] * routes_vessels[(v, r, t, s)]
                + gp.quicksum(
                    PORT_HANDLING[i, t]
                    * (
                        delivery_vessel[(i, v, r, t, w, s)]
                        + pickup_vessel[(i, v, r, t, w, s)]
                    )
                    for i in P_r[r]
                )
                for v in V
                for r in R_v[v]
            )
            for t in T_star
            for w in W
            for s in S_star
        ).getValue()

        container_handling_cost += (52 / NUM_WEEKS) * gp.quicksum(
            PROB_SCENARIO.iloc[0, s]
            * gp.quicksum(
                PORT_HANDLING[i, t]
                * (
                    delivery_vessel[(i, v, r, t, w, s)]
                    + pickup_vessel[(i, v, r, t, w, s)]
                )
                for v in V
                for r in R_v[v]
                for i in P_r[r]
            )
            for t in T_star
            for w in W
            for s in S_star
        ).getValue()

    model.data.sailing_cost = sailing_cost
    model.data.container_handling_cost = container_handling_cost

    return


def _calculate_vessel_capex(model, opt_sol_idx):

    S = model.data.S
    V = model.data.V
    N_s = model.data.N_s

    PROB_SCENARIO = model.data.PROB_SCENARIO
    VESSEL_INVESTMENT = model.data.VESSEL_INVESTMENT

    vessels = model.data.vessels

    vessel_capex = gp.quicksum(
        PROB_SCENARIO.iloc[0, s]
        * gp.quicksum(
            gp.quicksum(
                VESSEL_INVESTMENT[v, n] * vessels[(v, n)][opt_sol_idx] for v in V
            )
            for n in N_s[s]
        )
        for s in S
    ).getValue()
    model.data.vessel_capex = vessel_capex

    return


def _calculate_port_capex(model, opt_sol_idx):

    S = model.data.S
    P = model.data.P
    N_s = model.data.N_s

    PROB_SCENARIO = model.data.PROB_SCENARIO
    PORT_INVESTMENT = model.data.PORT_INVESTMENT

    ports = model.data.ports

    port_capex = 0
    port_capex = gp.quicksum(
        PROB_SCENARIO.iloc[0, s]
        * gp.quicksum(
            gp.quicksum(PORT_INVESTMENT[i, n] * ports[(i, n)][opt_sol_idx] for i in P)
            for n in N_s[s]
        )
        for s in S
    ).getValue()

    model.data.port_capex = port_capex

    return


def _calculate_truck_opex(model, opt_sol_idx):

    N = model.data.N
    W = model.data.W
    K = model.data.K
    T_n = model.data.T_n
    S_n = model.data.S_n
    P_k = model.data.P_k

    NUM_WEEKS = model.data.NUM_WEEKS
    PROB_SCENARIO = model.data.PROB_SCENARIO
    TRUCK_COST = model.data.TRUCK_COST
    PORT_CUSTOMER_DISTANCES = model.data.PORT_CUSTOMER_DISTANCES

    truck_opex = 0
    truck_distance = 0

    for n in N:
        delivery_truck = model.sp_data[n].delivery_truck[opt_sol_idx]
        pickup_truck = model.sp_data[n].pickup_truck[opt_sol_idx]

        T_star = T_n[n]
        S_star = S_n[n]

        truck_opex += (52 / NUM_WEEKS) * gp.quicksum(
            PROB_SCENARIO.iloc[0, s]
            * TRUCK_COST[i, k, t, s]
            * (delivery_truck[(i, k, t, w, s)] + pickup_truck[(i, k, t, w, s)])
            for k in K
            for i in P_k[k]
            for t in T_star
            for w in W
            for s in S_star
        ).getValue()

        truck_distance += (52 / NUM_WEEKS) * gp.quicksum(
            PROB_SCENARIO.iloc[0, s]
            * PORT_CUSTOMER_DISTANCES.iloc[i, k]
            * (delivery_truck[(i, k, t, w, s)] + pickup_truck[(i, k, t, w, s)])
            for k in K
            for i in P_k[k]
            for t in T_star
            for w in W
            for s in S_star
        ).getValue()

    model.data.truck_opex = truck_opex
    model.data.truck_distance = truck_distance
    return


def run_economic_analysis(model):
    opt_sol_idx = _find_best_iteration(model)

    _calculate_truck_opex(model, opt_sol_idx)
    _calculate_vessel_opex(model, opt_sol_idx)
    _calculate_port_capex(model, opt_sol_idx)
    _calculate_vessel_capex(model, opt_sol_idx)

    return
