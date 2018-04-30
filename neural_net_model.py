import networkx as nx
import numpy as np
import pandas as pd

from keras.models import Sequential
from collections import OrderedDict
from sklearn import preprocessing
from keras.layers import Dense

from graph_utils import get_attributes


def graph_to_dataframe(graph, remove_target_col=False):
    """
    Used for:
        - learning model
        - predicting while generating new graph
    """
    adj_matrix = nx.adjacency_matrix(graph)
    idxs = range(adj_matrix.shape[0])
    rows = []
    for node1_id in idxs:
        attrs1 = get_attributes(graph.node[node1_id].items(), 'node1_')
        for node2_id in idxs:
            attrs2 = get_attributes(graph.node[node2_id].items(), 'node2_')
            # to ensure that num_of_conn will be the last key
            # which is important when creating dataframe (last column - target)
            row = OrderedDict()
            row.update(attrs1)
            row.update(attrs2)
            if not remove_target_col:
                row['num_of_conn'] = adj_matrix[node1_id, node2_id]
            rows.append(row)

    df = pd.DataFrame(rows)

    # all attributes are strings (object type)
    # try to convert them to numeric (ignore errors - string which cannot be converted)
    df = df.apply(pd.to_numeric, errors='ignore')

    # map categorical to integers (cat codes)
    # with assumption that object type columns is string categorical
    coltypes_dict = dict(df.dtypes)
    str_columns = [key for key in coltypes_dict if coltypes_dict[key] == 'object']
    for column in str_columns:
        df[column] = df[column].astype('category')
    num_of_nodes = graph.number_of_nodes()
    cat_columns = []
    for column in df.select_dtypes(['category']).columns:
        # if number of unique values in column is equal to number of nodes in graph
        # then it's id / name attribute which should be dropped
        if df[column].nunique() != num_of_nodes:
            cat_columns.append(column)
        else:
            df.drop(column, axis=1, inplace=True)
            print('dropped column {}'.format(column))
    # hot-one encoding for categorical features
    df = pd.get_dummies(df, columns=cat_columns)
    # reorder columns to ensure that target column is last
    # colnames are like node1_attrname, node2_attrname, num_of_conn
    # so alphabetical order is correct
    df = df.reindex_axis(sorted(df.columns), axis=1)
    # feature values normalization
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(df)
    df = pd.DataFrame(np_scaled)

    return df


def get_trained_model(graph):
    df = graph_to_dataframe(graph)
    # number of attributes without target
    number_of_attrs = len(df.columns) - 1

    # Split DF into X and y
    X_train = df.iloc[:, :number_of_attrs]
    y_train = df.iloc[:, number_of_attrs]

    # Create model
    # Set seed for model reproducibility
    np.random.seed(93)
    model = Sequential()
    model.add(Dense(units=number_of_attrs, input_dim=number_of_attrs, activation='sigmoid'))
    model.add(Dense(units=1))

    model.compile(loss='binary_crossentropy', optimizer='sgd')

    # Train model
    model.fit(X_train, y_train, epochs=10)
    return model
