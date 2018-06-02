import os

import networkx as nx
import pandas as pd

delimiter = '\t'

prepared_datsets_path = 'prepared_datasets'


def load_dataset_to_graph(dataset_dir):
    prepared_edge_list = os.path.join(dataset_dir, 'edge_list.csv')
    prepared_node_attributes = os.path.join(dataset_dir, 'node_attributes.csv')

    #       LOAD EDGES
    graph = nx.read_edgelist(prepared_edge_list, create_using=nx.DiGraph(), nodetype=int)

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

primary_school_path = os.path.join(prepared_datsets_path, 'primary_school')
g = load_dataset_to_graph(primary_school_path)
print('done')
import pdb; pdb.set_trace()

# Graph to dataframe

# get trained model

# recreate by priority rank