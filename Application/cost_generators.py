from math import isnan
import pandas as pd
import numpy as np
from pandas.core.indexes import base

def sailing_cost(ROUTE_SAILING_COST,V,R,T, DISCOUNT_FACTOR=1):
    #preallocate for the 3D sailing cost matrix
    SAILING_COST = np.zeros([len(V),len(R),len(T)])
    for v in V:
        for r in R:
            for t in T:
                SAILING_COST[v,r,t] = ROUTE_SAILING_COST.iloc[v,r]*DISCOUNT_FACTOR**t

    return SAILING_COST

def truck_cost(PORT_CUSTOMER_DISTANCES, CUSTOMER_DATA, PORT_DATA, CO2_SCALE_FACTOR, P, K, T, S, DISCOUNT_FACTOR = 1):
    SOCIAL_COSTS = pd.DataFrame(data = {'Partially populated areas':[0.24, 0.47, 1.63, 0.21, 0.55, 0.03], 
        'Urban areas':[0.24, 3.05, 2.39, 1.48, 0.55, 0.03]}, index = ['CO2', 'Local emissions', 'Noise pollution', 'Road congestion', 'Accidents', 'Infrastructure wear'])
    URBAN_SOCIAL_COSTS_EMISSION_RELATED = SOCIAL_COSTS['Urban areas'][['CO2', 'Local emissions']].sum()
    POPAREAS_SOCIAL_COSTS_EMISSION_RELATED = SOCIAL_COSTS['Partially populated areas'][['CO2', 'Local emissions']].sum()

    URBAN_SOCIAL_COSTS_OTHER = SOCIAL_COSTS['Urban areas'][['Noise pollution', 'Road congestion', 'Accidents', 'Infrastructure wear']].sum()
    POPAREAS_SOCIAL_COSTS_OTHER = SOCIAL_COSTS['Partially populated areas'][['Noise pollution', 'Road congestion', 'Accidents', 'Infrastructure wear']].sum()
    #preallocate for the 3D sailing cost matrix
    TRUCK_COST = np.zeros([len(P),len(K),len(T),len(S)])
    for i in P:
        sotrasambandet = 50*(i==0)
        for t in T:
            for s in S: 
                emission_penal = np.prod(CO2_SCALE_FACTOR.iloc[s,:t])
                for k in K:
                    distance = PORT_CUSTOMER_DISTANCES.iloc[i,k]
                    if PORT_DATA.iloc[i,0] in ['City', 'All'] and CUSTOMER_DATA.iloc[k,0] == 'Consumer': # Urban social road cost per km
                        social = URBAN_SOCIAL_COSTS_EMISSION_RELATED* emission_penal + URBAN_SOCIAL_COSTS_OTHER
                    else:
                        social = POPAREAS_SOCIAL_COSTS_EMISSION_RELATED * emission_penal + POPAREAS_SOCIAL_COSTS_OTHER
                    TRUCK_COST[i,k,t,s] = int((distance*(31 + social) + 750*(distance>0) + sotrasambandet)*(DISCOUNT_FACTOR**t)) #31 kr / km, 750 kr start-fee (1500 kr per roundtrip). Assuming full truck both ways (if not multiply with 2)
    return TRUCK_COST

def port_handling_cost(HANDLING_COST,P,T,DISCOUNT_FACTOR=1):
    PORT_HANDLING = np.zeros([len(P),len(T)])
    for i in P:
        for t in T: 
            PORT_HANDLING[i,t] = HANDLING_COST.iloc[i,0]*DISCOUNT_FACTOR**t #Handling cost in specific port in addition to Agotnes (since pickup/delivery from main terminal is always included)
    return PORT_HANDLING

def vessel_investment(VESSEL_DATA,YEAR_OF_NODE,V,N, DISCOUNT_FACTOR = 1):
    VESSEL_LIFETIME = 20
    #Lifetime of vessels and ports should be assumed to be 20 years, not the number of years 
    NEW_VESSEL_INVESTMENT = np.zeros([len(V), len(N)])
    OPEX_PERCENT = 0.1 #Calculating the OPEX of the ships as a percent of the ships CAPEX
    for v in V:
        baseline_cost = VESSEL_DATA.iloc[v,2]
        for n in N:
            year = YEAR_OF_NODE.iloc[0,n]
            opex_yearly = baseline_cost*OPEX_PERCENT
            opex_total = 0
            for y in range(year,20): 
                opex_total += opex_yearly*DISCOUNT_FACTOR**y
            usage_adjusted_baseline_cost = baseline_cost*(VESSEL_LIFETIME-year)/VESSEL_LIFETIME
            NEW_VESSEL_INVESTMENT[v,n] = usage_adjusted_baseline_cost*DISCOUNT_FACTOR**year + opex_total
            
    return NEW_VESSEL_INVESTMENT

def port_investment(PORT_INVESTMENT,YEAR_OF_NODE,P,N, DISCOUNT_FACTOR = 1):
    PORT_LIFETIME = 20
    #NB: if we rewrite this to a function, NUM_YEARS also has to be given 
    NEW_PORT_INVESTMENT = np.zeros([len(P), len(N)])

    for i in P:
        baseline_cost = PORT_INVESTMENT.iloc[i,0]
        for n in N:
            year = YEAR_OF_NODE.iloc[0,n]
            usage_adjusted_baseline_cost = baseline_cost*(PORT_LIFETIME-year)/PORT_LIFETIME
            NEW_PORT_INVESTMENT[i,n] = usage_adjusted_baseline_cost*DISCOUNT_FACTOR**year

    return NEW_PORT_INVESTMENT

