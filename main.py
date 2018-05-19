import networkx as nx

import pandas as pd

from graph import generate_graph, attach_graph_attributes, \
    get_graph_measurements, compare_graph_measurements, print_comparison_results

from recreate_graph import graph_to_dataframe, get_trained_model, \
    recreate_by_priority_rank

def minMax(x):
    return pd.Series(index=['min','max'],data=[x.min(),x.max()])

params = {'n': 50, 'k': 2, 'p': 0.15}

graph = generate_graph(nx.watts_strogatz_graph, params)
attach_graph_attributes(graph)


df = graph_to_dataframe(graph)

model = get_trained_model(df)

new_graph = recreate_by_priority_rank(graph, df, model)

graph_measurements = get_graph_measurements(graph)
new_graph_measurements = get_graph_measurements(new_graph)
x = compare_graph_measurements(graph_measurements, new_graph_measurements)
print_comparison_results(x)
#import pdb; pdb.set_trace()