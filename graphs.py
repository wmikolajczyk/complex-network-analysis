import os
import networkx as nx
import pandas as pd

delimiter = '\t'


def generate_graph(graph_func, params):
    # set seed
    seed = 93
    # generate graph based on passed function and params
    graph = graph_func(**params, seed=seed)
    return graph


def load_dataset_to_graph(dataset_dir):
    prepared_edge_list = os.path.join(dataset_dir, 'edge_list.csv')
    # LOAD EDGES
    # Weights are auto loaded {'weight': 1.0}
    graph = nx.read_edgelist(prepared_edge_list, create_using=nx.DiGraph(), nodetype=int)
    return graph


def attach_graph_attributes(graph):
    # get list of attributes for each node id
    degree_centralities = nx.degree_centrality(graph)
    betweenness_centralities = nx.betweenness_centrality(graph)
    closeness_centralities = nx.closeness_centrality(graph)
    pageranks = nx.pagerank(graph)

    # attach appropriate attributes to each node
    for node_id in graph.nodes:
        node_attributes = {
            'degree_centrality': degree_centralities[node_id],
            'betweenness_centrality': betweenness_centralities[node_id],
            'closeness_centrality': closeness_centralities[node_id],
            'pagerank': pageranks[node_id]
        }
        graph.node[node_id].update(node_attributes)


def attach_real_attributes(graph, dataset_dir):
    prepared_node_attributes = os.path.join(dataset_dir, 'node_attributes.csv')
    # LOAD ATTRIBUTES
    attributes_data = pd.read_csv(prepared_node_attributes, delimiter=delimiter)
    # list of node attributes without node_id
    attributes_columns = list(attributes_data.columns)
    attributes_columns.remove('node_id')
    for node_id in graph.nodes:
        attrs = attributes_data.loc[attributes_data['node_id'] == node_id]
        node_attributes = {
            colname: attrs[colname].values[0]
            for colname in attributes_columns
        }
        graph.node[node_id].update(node_attributes)
    return graph
