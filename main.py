import networkx as nx

import pandas as pd

from graph import generate_graph, attach_graph_attributes, \
    get_graph_measurements, compare_graph_measurements

from recreate_graph import graph_to_dataframe, get_trained_model, \
    recreate_by_priority_rank


def minMax(x):
    return pd.Series(index=['min','max'],data=[x.min(),x.max()])


params = {'n': 30, 'm': 1}

graph = generate_graph(nx.barabasi_albert_graph, params)
attach_graph_attributes(graph)
graph_measurements = get_graph_measurements(graph)
x = compare_graph_measurements(graph_measurements, graph_measurements)


df = graph_to_dataframe(graph)

model = get_trained_model(df)

new_graph = recreate_by_priority_rank(graph, df, model)

#import pdb; pdb.set_trace()