import pandas as pd
import numpy as np


def generate_subsets(KEY_LIST, PORTS):
    subset_dict = dict.fromkeys(KEY_LIST)
    for key in KEY_LIST:
        subset = []
        subset.append(0)

        for i in range(1, len(PORTS)):
            if PORTS[i] == key:
                subset.append(i)

        subset_dict[key] = subset
    return subset_dict


def generate_routes(PORT_SUBSETS, PORT_DISTANCES, MAX_PORT_VISITS):
    routes = []
    distances = []
    num_visits = []

    for port_subset in PORT_SUBSETS:

        for port1 in port_subset[1:]:
            route = [0, port1, 0, 0, 0]
            if route not in routes and route[::-1] not in routes:
                routes.append(route)
                distances.append(route_distance(route, PORT_DISTANCES))
                num_visits.append(1)
            if MAX_PORT_VISITS > 1:
                for port2 in port_subset[1:]:
                    if port1 < port2:
                        route = [0, port1, port2, 0, 0]
                        if route not in routes and route[::-1] not in routes:
                            routes.append(route)
                            distances.append(route_distance(route, PORT_DISTANCES))
                            num_visits.append(2)

                        if MAX_PORT_VISITS > 2:
                            for port3 in port_subset[1:]:
                                if port2 < port3:
                                    candidate = []
                                    candidate.append([0, port1, port2, port3, 0])
                                    candidate.append([0, port1, port3, port2, 0])
                                    candidate.append([0, port2, port1, port3, 0])

                                    # Excludes all routes that are opposite direction. E.g. 0-1-2-3-0 is equal to 0-3-2-1-0.
                                    # This means that we do not care whether demand / pick-up is small / large in end of route.

                                    shortestroute = 10000
                                    for i in range(len(candidate)):
                                        newdist = route_distance(
                                            candidate[i], PORT_DISTANCES
                                        )
                                        if newdist < shortestroute:
                                            shortestroute = newdist
                                            routenumber = i

                                    route = candidate[routenumber]
                                    if (
                                        route not in routes
                                        and route[::-1] not in routes
                                    ):
                                        routes.append(route)
                                        distances.append(
                                            route_distance(route, PORT_DISTANCES)
                                        )
                                        num_visits.append(3)

    return routes, distances, num_visits


def route_distance(route, PORT_DISTANCES):
    dist = 0
    route = iter(route)
    prev_port = next(route)
    next_port = next(route)
    while True:
        dist += PORT_DISTANCES.iloc[prev_port, next_port]
        if next_port == 0:
            break
        try:
            prev_port = next_port
            next_port = next(route)
        except:
            break
    return dist


def distance2time(DISTANCE, NUM_VISITS, sailing_speed, handling_time):
    route_time = DISTANCE / sailing_speed + NUM_VISITS * handling_time
    return int(route_time)


def distance2time_cost(
    DISTANCE, NUM_VISITS, vessel_capacity, sailing_speed, handling_time
):
    route_time = int(DISTANCE / sailing_speed + NUM_VISITS * handling_time)

    k = 0.05
    alpha = 0.5
    beta = 3

    power_price = 0.8  # As in report

    # Regressionsformel fra marin. Alpha ~ 0.5. Beta ~ 3, k ~0.05
    power_need = k * (vessel_capacity ** alpha) * (sailing_speed ** beta)
    sailing_cost = int(power_price * power_need * route_time)

    return route_time, sailing_cost


def preprocess_routes(INSTANCE, MAX_PORT_VISITS):

    PORT_DISTANCES = pd.read_csv(
        f"Data/Instances/{INSTANCE}/Input_data/Port_Distances.csv", index_col=0
    )
    PORT_SUBSET = pd.read_csv(
        f"Data/Instances/{INSTANCE}/Input_data/Port_Data.csv", index_col=0
    ).loc[:, "Subset"]
    VESSEL_DATA = pd.read_csv(
        f"Data/Instances/{INSTANCE}/Input_data/Vessel_Data.csv", index_col=0
    )

    subset_names = PORT_SUBSET.unique()[1:]
    subset_dict = generate_subsets(subset_names, PORT_SUBSET)

    if "All" in subset_names and len(subset_names) > 2:
        all_subset = subset_dict["All"][1:]
        del subset_dict["All"]
        for s in subset_dict.keys():
            subset_dict[s] += all_subset

    ROUTES, DISTANCES, NUM_VISITS = generate_routes(
        subset_dict.values(), PORT_DISTANCES, MAX_PORT_VISITS
    )

    routes_time_dict = {}
    routes_cost_dict = {}
    idx = 0
    for s, c in zip(VESSEL_DATA["Speed"].tolist(), VESSEL_DATA["Capacity"].tolist()):
        handling_time = capacity2handlingTime(c)

        routes_time_dict[idx] = [
            distance2time_cost(distance, num_visits, c, s, handling_time)[0]
            for (distance, num_visits) in zip(DISTANCES, NUM_VISITS)
        ]
        routes_cost_dict[idx] = [
            distance2time_cost(distance, num_visits, c, s, handling_time)[1]
            for (distance, num_visits) in zip(DISTANCES, NUM_VISITS)
        ]

        idx += 1

    allroutes_df = pd.DataFrame(data=ROUTES).transpose()
    route_time_df = pd.DataFrame(data=routes_time_dict).transpose()
    route_cost_df = pd.DataFrame(data=routes_cost_dict).transpose()
    # HARD CODED
    vessel_route_feasibility_df = pd.DataFrame.from_dict(
        dict.fromkeys(["0", "1", "2"], np.ones(len(ROUTES))), orient="index"
    )  # All vessels can sail all routes

    allroutes_df.to_csv(f"Data/Instances/{INSTANCE}/Generated_data/Routes.csv")
    route_time_df.to_csv(
        f"Data/Instances/{INSTANCE}/Generated_data/Route_Sailing_Time.csv"
    )
    route_cost_df.to_csv(
        f"Data/Instances/{INSTANCE}/Generated_data/Route_Sailing_Cost.csv"
    )
    vessel_route_feasibility_df.to_csv(
        f"Data/Instances/{INSTANCE}/Generated_data/Route_Feasibility.csv"
    )

    return


def capacity2handlingTime(capacity):
    return capacity * 0.1 / 2
