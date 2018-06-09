import pandas as pd

from collections import OrderedDict
from sklearn import preprocessing


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
