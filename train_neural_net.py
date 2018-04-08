import networkx as nx
import numpy as np
import pandas as pd

from keras.models import Sequential
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
            row = {
                'num_of_conn': adj_matrix[node1_id, node2_id]
            }
            row.update(attrs1)
            row.update(attrs2)
            rows.append(row)
    return rows


def get_trained_model(graph):
    graph_data = graph_to_training_set(graph)
    df = pd.DataFrame(graph_data)
    # Split DF into X and y
    X_train = df.iloc[:, :8]
    y_train = df.iloc[:, 8]
    # Create model
    # Set seed for model recurrency
    np.random.seed(93)
    model = Sequential()
    model.add(Dense(units=8, input_dim=8, activation='sigmoid'))
    model.add(Dense(units=1))

    model.compile(loss='binary_crossentropy', optimizer='sgd')

    # Train model
    model.fit(X_train, y_train, epochs=100, verbose=0)
    return model
