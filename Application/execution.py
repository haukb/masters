#Standard library imports

#Special library imports 
from numpy.testing._private.utils import runstring
import pandas as pd
import dill

#Local imports 
from Versions.mp_2optimality import MP_2opt
from Versions.mp_branch_and_cut import MP_BBC
from Versions.mp_parallel_subproblems import MP_parallelSPs

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

def run_pipeline():
    pipeline = pd.read_csv('Execution/Pipeline.csv')
    run_ids = pd.read_csv('Execution/Run_ids.csv', index_col=0)
    rows, cols = pipeline.shape
    arg_names = [str.split(pipeline.columns[c], ' ')[0] for c in range(1,cols)]
    for i in range(rows):
        algo = pipeline.iloc[0,0]
        parameters = pipeline.iloc[0,1:]
        kwargs = make_kwargs(parameters,arg_names)
        if algo == 'MP_2opt':
            mp = MP_2opt(**kwargs)
        elif algo == 'MP_parallelSPs':
            mp = MP_parallelSPs(**kwargs)
        elif algo == 'MP_BBC':
            mp = MP_BBC
        else: 
            raise ValueError('Invalid algorithm name')

        #Solve the constructed master problem
        mp.solve()

        new_index = [run_ids.index[0]+1]
        run_ids = pd.concat([pd.DataFrame(index=new_index),run_ids], sort = False)
        run_ids.iloc[0] = pipeline.iloc[0]

        with open(f'Results/{run_ids.index[0]}.obj', "wb") as dill_file:
            dill.dump(mp.data, dill_file)

        #Remove executed run from pipeline
        pipeline = pipeline.iloc[1:]

    #Save the updated dataframes to csv files
    run_ids.to_csv('Execution/Run_ids.csv')
    #pipeline.to_csv('Execution/Pipeline.csv', index = False)

    return