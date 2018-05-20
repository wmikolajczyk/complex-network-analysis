import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

from graph import generate_graph, attach_graph_attributes, \
    get_graph_measurements, compare_graph_measurements, print_comparison_results

from recreate_graph import graph_to_dataframe, get_trained_model, \
    recreate_by_priority_rank


def minMax(x):
    return pd.Series(index=['min','max'],data=[x.min(),x.max()])

def draw_graph(graph):
    nx.draw(graph, with_labels=True)
    plt.show()


params = {'n': 100, 'm': 2}

graph = generate_graph(nx.barabasi_albert_graph, params)
attach_graph_attributes(graph)


df = graph_to_dataframe(graph)

model = get_trained_model(df)

new_graph = recreate_by_priority_rank(graph, df, model)

graph_measurements = get_graph_measurements(graph)
new_graph_measurements = get_graph_measurements(new_graph)
x = compare_graph_measurements(graph_measurements, new_graph_measurements)
print_comparison_results(x)

plt.figure(1)
nx.draw(graph, with_labels=True)
plt.figure(2)
nx.draw_shell(new_graph, with_labels=True)
plt.show()
#import pdb; pdb.set_trace()