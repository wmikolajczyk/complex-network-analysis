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


def recreate_real_graph(prepared_dataset_dir):
    print('Loading graph...')
    graph = load_dataset_to_graph(prepared_dataset_dir)
    # for x in range(301,1005):
    #     graph.remove_node(x)
    # attach graph attrs
    print('Attaching graph attributes...')
    attach_graph_attributes(graph)
    # attach real attrs
    print('Attaching real attributes...')
    attach_real_attributes(graph, prepared_dataset_dir)
    # graph to df
    print('Converting to dataframe...')
    df = graph_to_dataframe(graph)
    # train model
    print('Training model...')
    model = get_trained_model(df, epochs=4)
    # generate graph
    print('Recreating graph...')
    new_graph = recreate_by_priority_rank(graph, df, model)
    # compare
    print('Comparing graphs...')
    graph_measurements = get_graph_measurements(graph)
    new_graph_measurements = get_graph_measurements(new_graph)
    comparison = compare_graph_measurements(graph_measurements, new_graph_measurements)
    print_comparison_results(comparison)

    import pdb; pdb.set_trace()


primary_school_path = os.path.join(prepared_datasets_path, 'primary_school')
workplace_path = os.path.join(prepared_datasets_path, 'workplace')
highschool_2011_path = os.path.join(prepared_datasets_path, 'highschool_2011')
highschool_2012_path = os.path.join(prepared_datasets_path, 'highschool_2012')
hospital_path = os.path.join(prepared_datasets_path, 'hospital')
moreno_blogs_path = os.path.join(prepared_datasets_path, 'moreno_blogs')
moreno_sheep_path = os.path.join(prepared_datasets_path, 'moreno_sheep')
moreno_seventh_path = os.path.join(prepared_datasets_path, 'moreno_seventh')
# warning: big -> remove some nodes
petster_hamster_path = os.path.join(prepared_datasets_path, 'petster-hamster')
# warning: big -> remove some nodes
email_eu_path = os.path.join(prepared_datasets_path, 'email-Eu')

recreate_real_graph(workplace_path)

print('done')
