import os

from graph import attach_graph_attributes
from real_graphs import attach_real_attributes
from recreate_graph import graph_to_training_dataframe, preprocess_dataframe


def prepare_dataframes(df_dirpath, graph, dataset_path, delimiter):
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
