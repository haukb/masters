import pandas as pd

def port_customer_feasibility_dataframe_generator(PORT2CUSTOMER_DISTANCES):
    NUM_PORTS = PORT2CUSTOMER_DISTANCES.shape[0]
    NUM_CUSTOMERS = PORT2CUSTOMER_DISTANCES.shape[1]
    port_customer_feasibility_dict = dict.fromkeys(range(NUM_PORTS), [])
    for port in range(0, NUM_PORTS):
        port_customer_feasibility = []
        for customer in range(NUM_CUSTOMERS):
            main_port_dist = PORT2CUSTOMER_DISTANCES.iloc[0,customer]
            spoke_port_dist = PORT2CUSTOMER_DISTANCES.iloc[port,customer]

            if spoke_port_dist != -1 and spoke_port_dist <= main_port_dist:
                port_customer_feasibility.append(customer)
                
        port_customer_feasibility_dict[port] = port_customer_feasibility
    df = pd.DataFrame.from_dict(port_customer_feasibility_dict, dtype=object, orient='index').transpose()

    return df

def preprocess_feasibility(instance):

    try:
        PORT2CUSTOMER_DISTANCES = pd.read_csv(f'TestData/{instance}/Input_data/Port_Customer_Distances.csv', index_col=0)
        df = port_customer_feasibility_dataframe_generator(PORT2CUSTOMER_DISTANCES)
        df.to_csv(f'TestData/{instance}/Generated_data/Port_Customer_Feasibility.csv')
    except:
        print('Could not make port customer feasibility dataframe')

    return 