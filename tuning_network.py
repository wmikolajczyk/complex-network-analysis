import os
import random
import pandas as pd

from graph import attach_graph_attributes, get_graph_measurements,\
    compare_graph_measurements, print_comparison_results
from real_graphs import load_dataset_to_graph, attach_real_attributes
from recreate_graph import graph_to_training_dataframe, preprocess_dataframe,\
    get_trained_model, recreate_by_priority_rank

dataset_name = 'petster-hamster'

delimiter = '\t'
prepared_datasets_path = 'prepared_datasets'
prepared_dfs_path = 'prepared_training_dfs'

dataset_path = os.path.join(prepared_datasets_path, dataset_name)
df_dirpath = os.path.join(prepared_dfs_path, dataset_name)

# GET ORIGINAL GRAPH (FOR COMPARISON)
graph = load_dataset_to_graph(dataset_path)

# if there is too much nodes - remove
max_nodes = 250
overlimit_nodes = graph.number_of_nodes() - max_nodes
if overlimit_nodes > 0:
    print('Cutting nodes up to {}'.format(max_nodes))
    random.seed(93)
    nodes_to_remove = random.sample(graph.nodes(), overlimit_nodes)
    graph.remove_nodes_from(nodes_to_remove)

# PREPARE CSV FILE WITH DATAFRAME IF NOT EXISTS
if not os.path.exists(df_dirpath):
    os.mkdir(df_dirpath)
    # with graph attrs
    graph_attrs_path = os.path.join(df_dirpath, 'graph_attrs.csv')

    attach_graph_attributes(graph)

    df = graph_to_training_dataframe(graph)
    df = preprocess_dataframe(df, graph.number_of_nodes())
    df.to_csv(graph_attrs_path, sep=delimiter, index=False)
    # clear data
    for node in graph:
        keys = list(graph.nodes[node].keys())
        for key in keys:
            del graph.nodes[node][key]
    # with real attrs
    real_attrs_path = os.path.join(df_dirpath, 'real_attrs.csv')

    attach_real_attributes(graph, dataset_path)

    df = graph_to_training_dataframe(graph)
    df = preprocess_dataframe(df, graph.number_of_nodes())
    df.to_csv(real_attrs_path, sep=delimiter, index=False)
    # clear data
    for node in graph:
        keys = list(graph.nodes[node].keys())
        for key in keys:
            del graph.nodes[node][key]
    # with graph and real attrs
    graph_real_attrs_path = os.path.join(df_dirpath, 'graph_real_attrs.csv')

    attach_graph_attributes(graph)
    attach_real_attributes(graph, dataset_path)

    df = graph_to_training_dataframe(graph)
    df = preprocess_dataframe(df, graph.number_of_nodes())
    df.to_csv(graph_real_attrs_path, sep=delimiter, index=False)
    

    
    

# LOAD DATAFRAME
for df_path in ['graph_attrs.csv', 'real_attrs.csv', 'graph_real_attrs.csv']:
    path = os.path.join(df_dirpath, df_path)

    df = pd.read_csv(path, delimiter=delimiter)
    print('Training model...')
    model = get_trained_model(df, epochs=4, batch_size=64)


# print('Recreating by priority rank...')
# recreated_graph = recreate_by_priority_rank(graph, df, model)
# print('Getting original graph measurements...')
# graph_measurements = get_graph_measurements(graph)
# print('Getting recreated graph measurements')
# recreated_graph_measurements = get_graph_measurements(recreated_graph)
# print('Making comparison...')
# comparison = compare_graph_measurements(graph_measurements, recreated_graph_measurements)
# print_comparison_results(comparison)

import pdb; pdb.set_trace()
