import networkx as nx
import matplotlib.pyplot as plt

from generate_graph_nn import get_trained_model, generate_graph_by_nn


def attach_attributes(graph):
    degree_centralities = nx.degree_centrality(graph)
    betweenness_centralities = nx.betweenness_centrality(graph)
    closeness_centralities = nx.closeness_centrality(graph)
    pageranks = nx.pagerank(graph)

    for node_id in graph.nodes:
        node_attributes = {
            'degree_centrality': degree_centralities[node_id],
            'betweenness_centrality': betweenness_centralities[node_id],
            'closeness_centrality': closeness_centralities[node_id],
            'pagerank': pageranks[node_id]
        }
        graph.node[node_id].update(node_attributes)


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


def draw_graph(graph):
    nx.draw(graph, with_labels=True)
    plt.show()
