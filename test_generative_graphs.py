import networkx as nx
import matplotlib.pyplot as plt

from compare_graphs import compare
from generate_graph_nn import attach_attributes, get_trained_model, generate_graph_by_nn

def test_generative_graph(graph_func, *args, **kwargs):
    seed = 93

    # GENERATE GRAPH, CALCULATE NODE ATTRIBUTES AND ATTACH THESE ATTRIBUTES
    graph = graph_func(*args, seed=seed)
    attach_attributes(graph)
    # CALCULATE AVG NUM EDGES FOR PRIORITY RANK, TRAIN MODEL, GENERATE NEW GRAPH BY NN USING TRAINED MODEL
    avg_num_edges = round(graph.number_of_edges() / graph.number_of_nodes())
    model = get_trained_model(graph)
    new_graph = generate_graph_by_nn(model, graph, avg_num_edges)
    # GET COMPARSION DICT FOR ORIGINAL AND GENERATED GRAPHS
    comparsion_dict = compare(graph, new_graph)
    # PRINT ORIGINAL AND GENERATED GRAPH COMPARSION RESULTS
    print('Graph type: {type}'.format(type=graph_func.__name__))
    [print('{}: {}'.format(k, v)) for k, v in comparsion_dict.items()]

    if kwargs.get('show_graph'):
        # DRAW ORIGINAL GRAPH
        plt.figure(1)
        nx.draw(graph, with_labels=True)
        # DRAW GENERATED GRAPH
        plt.figure(2)
        nx.draw(new_graph, with_labels=True)
        plt.show()

# Probability of connection between nodes
erdos_renyi_params = [
    {'n': 30, 'p': 0.05},
    {'n': 30, 'p': 0.1},
    {'n': 30, 'p': 0.3}
]
for param in erdos_renyi_params:
    test_generative_graph(nx.erdos_renyi_graph, param['n'], param['p'], show_graph=True)

# Probability of
watts_strogatz_params = [
    {'n': 30, 'k': 2, 'p': 0.05},
    {'n': 30, 'k': 2, 'p': 0.1},
    {'n': 30, 'k': 2, 'p': 0.3}
]
for param in watts_strogatz_params:
    test_generative_graph(nx.watts_strogatz_graph, param['n'], param['k'], param['p'], show_graph=True)

barabasi_albert_params = [
    {'n': 30, 'm': 1},
    {'n': 30, 'm': 2},
    {'n': 30, 'm': 5}
]
for param in barabasi_albert_params:
    test_generative_graph(nx.barabasi_albert_graph, param['n'], param['m'], show_graph=True)
