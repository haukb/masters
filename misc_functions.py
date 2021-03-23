import pandas as pd

def get_same_year_nodes(node, N, YEAR_OF_NODE):
    year = YEAR_OF_NODE.iloc[0,node]
    nodes = []
    for n in N:
        if year == YEAR_OF_NODE.iloc[0,n]:
            nodes.append(n)

    return nodes