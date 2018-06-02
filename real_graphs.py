import os

import networkx as nx
import pandas as pd

from graph import attach_graph_attributes, get_graph_measurements, compare_graph_measurements, print_comparison_results
from recreate_graph import graph_to_dataframe, get_trained_model, recreate_by_priority_rank

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
graph = load_dataset_to_graph(primary_school_path)
# attach graph attrs
attach_graph_attributes(graph)
# graph to df
df = graph_to_dataframe(graph)
# train model
model = get_trained_model(df, epochs=4)
# SET WEIGHTS WHEN CREATING 
# generate graph
new_graph = recreate_by_priority_rank(graph, df, model)
# compare
graph_measurements = get_graph_measurements(graph)
new_graph_measurements = get_graph_measurements(new_graph)
comparison = compare_graph_measurements(graph_measurements, new_graph_measurements)
print_comparison_results(comparison)

print('done')
import pdb; pdb.set_trace()