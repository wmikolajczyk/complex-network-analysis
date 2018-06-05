import os

import networkx as nx
import pandas as pd

from graph import attach_graph_attributes, get_graph_measurements, compare_graph_measurements, print_comparison_results
from recreate_graph import graph_to_dataframe, get_trained_model, recreate_by_priority_rank

delimiter = '\t'

prepared_datasets_path = 'prepared_datasets'


def attach_real_attributes(graph, dataset_dir):
    prepared_node_attributes = os.path.join(dataset_dir, 'node_attributes.csv')
    #       LOAD ATTRIBUTES
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

def load_dataset_to_graph(dataset_dir):
    prepared_edge_list = os.path.join(dataset_dir, 'edge_list.csv')
    #       LOAD EDGES
    # Weights are auto loaded {'weight': 1.0}
    graph = nx.read_edgelist(prepared_edge_list, create_using=nx.DiGraph(), nodetype=int)
    return graph
