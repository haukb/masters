import numpy as np
import pandas as pd
from utils.misc_functions import get_obj


def _average_list(lists):
    length = len(lists[0])
    iterations = len(lists)
    average = np.zeros(length)
    for i in range(length):
        for j in range(iterations):
            average[i] += lists[j][i]
        average[i] /= iterations
    return average


def variability_index(lists):
    average = _average_list(lists)
    length = len(lists[0])
    iterations = len(lists)
    variability = np.zeros([iterations, length])
    index_values = np.zeros(iterations)
    for j in range(iterations):
        for i in range(length):
            variability[j, i] += average[i] - lists[j][i]
            if variability[j, i] < 0:
                variability[j, i] *= -1

    for j in range(iterations):
        index_values[j] = sum(variability[j, :]) / sum(average)

    return index_values


def _get_port_decisions(run_ids):
    port_decisions = []
    for i in run_ids:
        mp_data = get_obj(i)
        l = [0 for i in mp_data.P]
        for port in mp_data.P:
            l[port] = mp_data.ports[port, 0][-1]
        port_decisions.append(l)
    return port_decisions


def _get_vessel_decisions(run_ids):
    vessel_decisions = []
    for i in run_ids:
        mp_data = get_obj(i)
        l = [0 for v in mp_data.V]
        for vessel in mp_data.V:
            l[vessel] = mp_data.vessels[vessel, 0][-1]
        vessel_decisions.append(l)
    return vessel_decisions


def variability_from_runs(run_ids):
    vessels = _get_vessel_decisions(run_ids)
    ports = _get_port_decisions(run_ids)
    vessels_var = variability_index(vessels)
    ports_var = variability_index(ports)
    return {"Ports variability": ports_var, "Vessels variability": vessels_var}


def timeVgap(run_id):
    run = get_obj(run_id)

    if run.warm_start_solve_time > 0:
        warm_start = True
    else:
        warm_start = False

    if warm_start is True:
        mp_time = [0 for n in range(len(run.mp_solve_time) + 1)]
        sp_time = mp_time.copy()
        gap = mp_time.copy()
    else:
        mp_time = [0 for n in range(len(run.mp_solve_time))]
        sp_time = mp_time.copy()
        gap = mp_time.copy()

    if warm_start is False:
        for i in range(len(run.mp_solve_time)):
            mp_time[i] = round(run.mp_solve_time[i], 1)
            sp_time[i] = round(run.sp_solve_time[i], 1)
            try:
                gap[i] = round(
                    (min(run.upper_bounds[:i]) - run.lower_bounds[i])
                    / run.lower_bounds[i],
                    4,
                )
            except:
                gap[i] = 1000
    else:
        for j in range(len(run.mp_solve_time) + 1):
            if j == 0:
                mp_time[j] = round(run.warm_start_solve_time, 1)
            else:
                mp_time[j] = round(run.mp_solve_time[j - 1], 1)
                sp_time[j] = round(run.sp_solve_time[j - 1], 1)
                try:
                    gap[j] = round(
                        (min(run.upper_bounds[:j]) - run.lower_bounds[j])
                        / run.lower_bounds[j],
                        4,
                    )
                except:
                    gap[j] = 1000

    df = pd.DataFrame(data={"mp_time": mp_time, "sp_time": sp_time, "gap": gap})

    return df


def save_results2csv(ids):
    for i in ids:
        df = timeVgap(run_id=i)
        df.to_csv(f"OUTPUTS/{i}.csv")
    return
