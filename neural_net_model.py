import networkx as nx
import numpy as np
import pandas as pd

from keras.models import Sequential
from collections import OrderedDict

from keras.layers import Dense

from graph_utils import get_attributes


def graph_to_training_set(graph):
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
            row['num_of_conn'] = adj_matrix[node1_id, node2_id]
            rows.append(row)
    return rows


def get_trained_model(graph):
    graph_data = graph_to_training_set(graph)
    df = pd.DataFrame(graph_data)
    # number of attributes without target
    number_of_attrs = len(df.columns) - 1

    # map categorical to integers (cat codes)
    # with assumption that object type columns is string categorical
    coltypes_dict = dict(df.dtypes)
    str_columns = [key for key in coltypes_dict if coltypes_dict[key] == 'object']
    for column in str_columns:
        df[column] = df[column].astype('category')
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

    # TODO: normalize!

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
