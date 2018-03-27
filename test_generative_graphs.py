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
