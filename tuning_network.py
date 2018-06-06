import os
import pandas as pd

from graph import attach_graph_attributes, get_graph_measurements,\
    compare_graph_measurements, print_comparison_results
from real_graphs import load_dataset_to_graph, attach_real_attributes
from recreate_graph import graph_to_training_dataframe, preprocess_dataframe,\
    get_trained_model, recreate_by_priority_rank

dataset_name = 'primary_school'

delimiter = '\t'
prepared_datasets_path = 'prepared_datasets'
prepared_dfs_path = 'prepared_training_dfs'

dataset_path = os.path.join(prepared_datasets_path, dataset_name)
df_path = os.path.join(prepared_dfs_path, dataset_name + '.csv')

# GET ORIGINAL GRAPH (FOR COMPARISON)
graph = load_dataset_to_graph(dataset_path)
# PREPARE CSV FILE WITH DATAFRAME
if not os.path.exists(df_path):
    attach_graph_attributes(graph)
    attach_real_attributes(graph, dataset_path)
    df = graph_to_training_dataframe(graph)
    df = preprocess_dataframe(df, graph.number_of_nodes())
    df.to_csv(df_path, sep=delimiter, index=False)
# LOAD DATAFRAME
df = pd.read_csv(df_path, delimiter=delimiter)
print('Training model...')
model = get_trained_model(df, epochs=4)
print('Recreating by priority rank...')
recreated_graph = recreate_by_priority_rank(graph, df, model)
print('Getting original graph measurements...')
graph_measurements = get_graph_measurements(graph)
print('Getting recreated graph measurements')
recreated_graph_measurements = get_graph_measurements(recreated_graph)
print('Making comparison...')
comparison = compare_graph_measurements(graph_measurements, recreated_graph_measurements)
print_comparison_results(comparison)
import pdb; pdb.set_trace()
