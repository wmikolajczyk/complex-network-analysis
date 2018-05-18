import numpy as np
import pandas as pd

from collections import OrderedDict

from keras.models import Sequential
from keras.layers import Dense


def recreate_graph(graph):
    # calculate average number of edges in original graph
    avg_num_edges = round(graph.number_of_edges() / graph.number_of_nodes())
    # build df form graph
    df = graph_to_dataframe(graph)
    # train a model
    model = get_trained_model(df)
    # generate new graph using trained model
    new_graph = generate_by_priority_rank(model, graph, avg_num_edges)
    return new_graph


def get_trained_model(df):
    # number of attributes without target
    number_of_attrs = len(df.columns) - 1

    # Split DF into X and y
    X_train = df.iloc[:, :number_of_attrs]
    y_train = df.iloc[:, number_of_attrs]

    # set seed for model reproducibility
    np.random.seed(93)

    model = Sequential()
    model.add(Dense(
        units=number_of_attrs,
        input_dim=number_of_attrs,
        activation='sigmoid'))
    model.add(Dense(units=1))

    model.compile(loss='binary_crossentropy', optimizer='sgd')
    model.fit(X_train, y_train, epochs=10)
    return model


def get_prefix_attributes(prefix, node):
    attributes_dict = {
        prefix + key: value
        for key, value in node.items()
    }
    return attributes_dict


def graph_to_dataframe(graph):
    rows = []
    for node1_id in graph.nodes:
        attrs1 = get_prefix_attributes('node1_', graph.node[node1_id])
        for node2_id in graph.nodes:
            attrs2 = get_prefix_attributes('node2_', graph.node[node2_id])
            row = OrderedDict()
            row.update(attrs1)
            row.update(attrs2)
            row['num_of_edges'] = graph.number_of_edges(node1_id, node2_id)
            rows.append(row)

    df = pd.DataFrame(rows)
    return df

# TODO:
# dataframe processing
# handle categorical columns (str)
# drop if amount of unique values = num of nodes -> id column
# get hot-one encoding (get_dummies)
# minmax scaler - normalize values
