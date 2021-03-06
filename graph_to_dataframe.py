import pandas as pd
import os

from collections import OrderedDict
from sklearn import preprocessing

from graphs import attach_graph_attributes, attach_real_attributes


def set_prefix_attributes(prefix, node):
    attributes_dict = {
        prefix + key: value
        for key, value in node.items()
    }
    return attributes_dict


def graph_to_dataframe(graph):
    nodes_with_attrs = [
        {
            'node_id': node_id,
            **attrs_dict
        }
        for node_id, attrs_dict in graph.nodes(data=True)
    ]
    df = pd.DataFrame(nodes_with_attrs)
    return df


def graph_to_training_dataframe(graph):
    rows = []
    for node1_id in graph.nodes:
        attrs1 = set_prefix_attributes('node1_', graph.node[node1_id])
        for node2_id in graph.nodes:
            attrs2 = set_prefix_attributes('node2_', graph.node[node2_id])
            row = OrderedDict()
            row.update(attrs1)
            row.update(attrs2)

            edge_data = graph.get_edge_data(node1_id, node2_id)
            if edge_data:
                weight = edge_data['weight']
            else:
                weight = 0
            row['num_of_edges'] = weight
            rows.append(row)
    df = pd.DataFrame(rows)
    return df


def preprocess_dataframe(df, number_of_nodes):
    # all attributes are strings (object type)
    # try to convert them to numeric (ignore errors - string which cannot be converted)
    df = df.apply(pd.to_numeric, errors='ignore')
    # handle categorical columns (str)
    coltypes_dict = dict(df.dtypes)
    # get string columns
    str_columns = [key for key in coltypes_dict
                   if coltypes_dict[key] == 'object']
    # set string columns to category
    for column in str_columns:
        df[column] = df[column].astype('category')
    # drop column if number of unique values = number of nodes
    #   because it means this column is like string id for node
    for column in df.select_dtypes(['category']).columns:
        if df[column].nunique() == number_of_nodes:  # graph.number_of_nodes():
            df.drop(column, axis=1, inplace=True)
            print('dropped column {}'.format(column))
    # get dummies for categorical variable
    df = pd.get_dummies(df, columns=df.select_dtypes(['category']).columns)
    # reorder columns to ensure that target column is last
    #   colnames are like node1_attrname, node2_attrname, num_of_conn
    #   so alphabetical order is correct
    df = df.reindex(sorted(df.columns), axis=1)

    # minmax scaler - normalize values
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled_values = min_max_scaler.fit_transform(df)
    df.loc[:, :] = scaled_values
    return df


def export_training_dataframes(graph, dataset_path, df_dirpath, delimiter='\t'):
    if not os.path.exists(df_dirpath):
        os.makedirs(df_dirpath)

    # no attrs - only num of edges normalized
    no_attrs_path = os.path.join(df_dirpath, 'no_attrs.csv')

    df = graph_to_training_dataframe(graph)
    df = preprocess_dataframe(df, graph.number_of_nodes())
    df.to_csv(no_attrs_path, sep=delimiter, index=False)
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
