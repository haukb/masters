import numpy as np


def max_vessels_heuristic(model) -> list:
    """Function to calculate the practical max number of the different
    vessel types needed to serve the demand.

    Args:
        model (Master_problem): Custom Master_problem class

    Returns:
        list: each position represent the max number of vessels needed
        of that type (e.g. 0,1,2)
    """
    max_vessels = np.ones(model.data.NUM_VESSELS, int)
    avg_weekly_demand = model.data.DELIVERY[:, 19, :, 0].sum() / model.data.NUM_WEEKS
    for v in model.data.V:
        num_trips_needed = avg_weekly_demand / model.data.VESSELS.Capacity[v]
        # If we want to be more conservative we can use the max(not mean) sailing time
        total_distribution_time = (
            num_trips_needed * model.data.ROUTE_SAILING_TIME.iloc[v, :].mean()
        )
        number_of_vessels_needed = (
            total_distribution_time / model.data.TIMEPERIOD_DURATION.iloc[0, 0]
        )
        max_vessels[v] = int(np.ceil(number_of_vessels_needed))

    return max_vessels
