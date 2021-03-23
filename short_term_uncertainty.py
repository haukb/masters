import numpy as np
import pandas as pd

# Function that takes a baseline demand as input and returns a demand drawns a new random demand from a standard distribution centered at the baseline demand, and where the standard deviation is equal to the expected demand 
def draw_weekly_demand(baseline_demand, customer_type):
    #0.64 is from the analysis of standard deviation calculations
    if customer_type == 'Consumer':
        random_demand = (np.random.normal(loc=baseline_demand,scale=baseline_demand*0.64))

    elif customer_type == 'Industrial':
        random_demand = (np.random.normal(loc=baseline_demand,scale=baseline_demand*0.1))
        
    return max(min(random_demand,baseline_demand*2),0) #ensures that demand can't be negative

def generate_random_weeks(instances):

    for inst in instances:
        CUSTOMER_DATA =  pd.read_csv(f'TestData/{inst}/Input_data/Customer_Data.csv', index_col=0)
        SCENARIOYEAR_GROWTH_FACTOR = pd.read_csv(f'TestData/{inst}/Input_data/ScenarioYear_Growth_Factor.csv', index_col=0)

        num_customers = CUSTOMER_DATA.shape[0]
        _, num_years = SCENARIOYEAR_GROWTH_FACTOR.shape
        num_weeks = 52 #here we use an arbitrary high value, but the model will only sample from the correct week number anyway

        K = np.arange(num_customers)
        T = np.arange(num_years)
        W = np.arange(num_weeks)

        weekly_variation_factor = np.zeros([num_customers, num_weeks, num_years])

        for c in K: 
            customer_type = CUSTOMER_DATA.iloc[c,0]
            for w in W:
                for t in T: 
                    weekly_variation_factor[c,w,t] = draw_weekly_demand(1, customer_type) 
        
        ravelled = np.ravel(weekly_variation_factor)
        filename = f'TestData/{inst}/Generated_data/Weekly_Variation_Factor.csv'
        ravelled.tofile(filename)
    
    return

if __name__ == '__main__':
    #instances = ['Full_instance'] #This line choses for which instance to generate new random weeks
    #Wathc out, do not run this file unless you want to replace the current random weeks
    
    generate_random_weeks(instances)