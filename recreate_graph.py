import networkx as nx
import numpy as np


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

from tensorflow import set_random_seed


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

    # init normal ?
    model.add(Dense(
        units=number_of_attrs,
        input_dim=number_of_attrs,
        activation='linear'))
    model.add(Dense(
        units=round(number_of_attrs / 2),
        input_dim=round(number_of_attrs / 2),
        activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer=SGD(lr=0.01), loss='mean_squared_error', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

    evaluation = model.evaluate(X_train, y_train, verbose=1)
    print('loss: {}, accuracy: {}'.format(evaluation[0], evaluation[1]))
    return model


def recreate_by_priority_rank(graph, df, model):
    num_edges = round(graph.number_of_edges() / graph.number_of_nodes())
    num_of_nodes = graph.number_of_nodes()

    new_graph = nx.DiGraph()
    # drop target column
    X_test = df.drop(['num_of_edges'], axis=1)
    # predict num_edges
    y_pred = model.predict(X_test)

    # used when calculating probability ranking
    harmonic_number = sum([
        1 / k for k in range(1, num_of_nodes + 1)
    ])
    for node1_index, node1_id in enumerate(graph.nodes):
        # get dict of node rankings
        #   [(node0_id, num_edges), (node1_id, num_edges)]
        similarities = []
        for node2_index, node2_id in enumerate(graph.nodes):
            node_index = node1_index * num_of_nodes + node2_index
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
        target_nodes = np.random.choice(ranking, size=num_edges,
                                        replace=False, p=probability)
        # Add edges to new graph
        # creating verticles while adding edges
        for target_node in target_nodes:
            new_graph.add_edge(node1_id, target_node)

    # import matplotlib.pyplot as plt
    # plt.figure(1)
    # plt.plot(y_pred)
    # plt.figure(2)
    # plt.plot(df[['num_of_edges']])
    # plt.show()
    return new_graph
