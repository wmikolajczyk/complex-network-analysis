import networkx as nx
import numpy as np
import pandas as pd

from collections import OrderedDict

from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing
from tensorflow import set_random_seed


def recreate_graph(graph):
    # build df form graph
    df = graph_to_dataframe(graph)
    # train a model
    model = get_trained_model(df)
    # generate new graph using trained model
    new_graph = recreate_by_priority_rank(graph, df, model)
    return new_graph


def get_trained_model(df, epochs=128, batch_size=64, verbose=1):
    # number of attributes without target
    number_of_attrs = len(df.columns) - 1

    # Split DF into X and y
    X_train = df.iloc[:, :number_of_attrs]
    y_train = df.iloc[:, number_of_attrs]

    # set seed for model reproducibility
    np.random.seed(93)
    # tensorflow random seed
    set_random_seed(2)

    model = Sequential()

    model.add(Dense(
        units=number_of_attrs,
        input_dim=number_of_attrs,
        activation='relu'))
    model.add(Dense(
        units=round(number_of_attrs / 2),
        input_dim=round(number_of_attrs / 2),
        activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='sgd')

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
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

    # all attributes are strings (object type)
    # try to convert them to numeric (ignore errors - string which cannot be converted)
    df = df.apply(pd.to_numeric, errors='ignore')
    # handle categorical columns (str)
    coltypes_dict = dict(df.dtypes)
    str_columns = [key for key in coltypes_dict
                   if coltypes_dict[key] == 'object']
    for column in str_columns:
        df[column] = df[column].astype('category')
    for column in df.select_dtypes(['category']).columns:
        if df[column].nunique() == graph.number_of_nodes():
            df.drop(column, axis=1, inplace=True)
            print('dropped column {}'.format(column))
    df = pd.get_dummies(df, columns=df.select_dtypes(['category']).columns)
    # reorder columns to ensure that target column is last
    # colnames are like node1_attrname, node2_attrname, num_of_conn
    # so alphabetical order is correct
    df = df.reindex(sorted(df.columns), axis=1)

    # minmax scaler - normalize values
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled_values = min_max_scaler.fit_transform(df)
    df.loc[:, :] = scaled_values
    return df


def recreate_by_priority_rank(graph, df, model):
    num_edges = round(graph.number_of_edges() / graph.number_of_nodes())
    num_of_nodes = graph.number_of_nodes()

    new_graph = nx.empty_graph(n=num_of_nodes)
    # drop target column
    X_test = df.drop(['num_of_edges'], axis=1)
    # predict num_edges
    y_pred = model.predict(X_test)

    # used when calculating probability ranking
    harmonic_number = sum([
        1 / k for k in range(1, num_of_nodes + 1)
    ])
    for node1_id in graph.nodes:
        # get dict of node rankings
        #   [(node0_id, num_edges), (node1_id, num_edges)]
        similarities = []
        for node2_id in graph.nodes:
            node_index = node1_id * num_of_nodes + node2_id
            similarities.append((node2_id, y_pred.item(node_index)))
        # Node ranking sorted in descending similarity order
        ranking = [
            node2_id
            for (node2_id, similarity) in
            sorted(similarities, key=lambda x: x[1], reverse=True)
        ]
        # Probability of connection to node on each position at the ranking
        probability = [
            1 / (harmonic_number * index)
            for index, _ in enumerate(ranking, start=1)
        ]
        # Choose randomly k (num_edges) nodes to make connections with
        sorted_sim = sorted(similarities, key=lambda x: x[1], reverse=True)
        # get edges amount from prediction
        # edge_thresh = max(y_pred).item() * 0.7
        # num_e = len([x for x in sorted_sim if x[1] > edge_thresh])
        target_nodes = np.random.choice(ranking, size=num_edges,
                                        replace=False, p=probability)
        # Add edges to new graph
        for target_node in target_nodes:
            new_graph.add_edge(node1_id, target_node)

    # import matplotlib.pyplot as plt
    # plt.figure(1)
    # plt.plot(y_pred)
    # plt.figure(2)
    # plt.plot(df[['num_of_edges']])
    # plt.show()
    return new_graph
