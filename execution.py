#Standard library imports
import os
from time import process_time

#Special library imports 
from numpy.testing._private.utils import runstring
import pandas as pd
import dill
from csv import writer

#Local imports 
from algorithms.mp_2optimality import MP_2opt
from algorithms.mp_branch_and_cut import MP_BBC
from algorithms.mp_parallel_subproblems import MP_parallelSPs

def make_kwargs(row, arg_names):
    kwargs = {}
    for c in range(len(row)):
        val = row[c]
        if not pd.isnull(val):
            kwargs[arg_names[c]] = val
    return kwargs

def make_id(run_ids, algo, kwargs):
    new_index = [run_ids.index[0]+1]
    run_ids = pd.concat([pd.DataFrame(index=[new_index]),run_ids], sort = False)
    run_ids.iloc[0] = kwargs
    run_ids.loc[new_index,'Algorithm name'] = algo
    return run_ids

def get_next_run():
    #Read the pipeline df
    pipeline = pd.read_csv('Execution/Pipeline.csv')
    row = pipeline.iloc[0]
    #Remove the line just read, and save the new df
    pipeline = pipeline.iloc[1:]
    pipeline.to_csv('Execution/Pipeline.csv', index = False)
    return row

def save_run(mp,row):
    #Read the run_ids df
    run_ids = pd.read_csv('Execution/Run_ids.csv', index_col=0)
    #add a new empty row on top
    new_index = [run_ids.index[0]+1]
    run_ids = pd.concat([pd.DataFrame(index=new_index),run_ids], sort = False)
    #fill the row with the current run
    run_ids.iloc[0] = row
    run_ids.to_csv('Execution/Run_ids.csv')
    with open(f'Results/{run_ids.index[0]}.obj', "wb") as dill_file:
            dill.dump(mp.data, dill_file)

    return

def run_pipeline():
    pipeline = pd.read_csv('Execution/Pipeline.csv')
    arg_names = [str.split(pipeline.columns[c], ' ')[0] for c in range(1,pipeline.shape[1])]
    while True: 
        #Check if there are any more runs in the pipeline
        try:
            row = get_next_run()
        except: 
            break
        algo = row[0]
        parameters = row[1:]
        kwargs = make_kwargs(parameters,arg_names)
        if algo == 'MP_2opt':
            mp = MP_2opt(**kwargs)
        elif algo == 'MP_parallelSPs':
            mp = MP_parallelSPs(**kwargs)
        elif algo == 'MP_BBC':
            mp = MP_BBC
        else: 
            raise ValueError('Invalid algorithm name')
        
        mp.solve()
        save_run(mp, row)
    
    return

if __name__ == '__main__':
    run_pipeline()