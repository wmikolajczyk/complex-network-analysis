import networkx as nx
import matplotlib.pyplot as plt

from compare_graphs import compare
from generate_graph_nn import attach_attributes, get_trained_model, generate_graph_by_nn


def generate_graph(graph_func, params):
    # set seed
    seed = 93
    # generate graph by passed function and args
    graph = graph_func(**params, seed=seed)
    # calculate and attach node attributes
    attach_attributes(graph)
    return graph

def recreate_graph(graph):
    # calculate average number of edges in original graph
    avg_num_edges = round(graph.number_of_edges() / graph.number_of_nodes())
    # train a model
    model = get_trained_model(graph)
    # generate new graph using trained model
    new_graph = generate_graph_by_nn(model, graph, avg_num_edges)
    return new_graph

def print_comparsion_result(comparsion_dict):
    for key, val in comparsion_dict.items():
        print('{}: {}'.format(key, val))


def draw_graph(graph):
    nx.draw(graph, with_labels=True)
    plt.show()
