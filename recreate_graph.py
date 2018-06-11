import random
import networkx as nx
import numpy as np

from keras.models import Sequential
from keras.layers import Dense

from tensorflow import set_random_seed


def get_model(number_of_attrs):
    """
    Number of units in layers (more than 1 unit in layer):
        1. first = number_of_attrs * 1.5
        2. second = first / 2 or number_of_attrs
        3. third = second / 2 or second
        4. 1
    """
    # set seed for model reproducibility
    np.random.seed(93)
    # tensorflow random seed
    set_random_seed(2)
    # calculate number of units in second and third layer
    #   in moreno_sheep dataset there are only 2 attributes
    #   so it's important to handle this case
    num_of_units_second_layer = round(number_of_attrs / 2)
    if not num_of_units_second_layer > 1:
        num_of_units_second_layer = number_of_attrs

    num_of_units_third_layer = round(num_of_units_second_layer / 2)
    if not num_of_units_third_layer > 1:
        num_of_units_third_layer = num_of_units_second_layer

    model = Sequential()

    model.add(Dense(
        units=round(number_of_attrs * 1.5),
        input_dim=number_of_attrs,
        activation='relu'
    ))
    model.add(Dense(
        units=num_of_units_second_layer,
        activation='relu'
    ))
    model.add(Dense(
        units=num_of_units_third_layer,
        activation='relu'
    ))
    model.add(Dense(
        units=1, activation='sigmoid')
    )
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    return model


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

    model = get_model(number_of_attrs)

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

    evaluation = model.evaluate(X_train, y_train, verbose=1)
    print('loss: {}, accuracy: {}'.format(evaluation[0], evaluation[1]))
    return model


def recreate_by_priority_rank(graph, target_col):
    # target_col - nd_array type
    num_edges = round(graph.number_of_edges() / graph.number_of_nodes())
    num_of_nodes = graph.number_of_nodes()

    new_graph = nx.DiGraph()

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
            similarities.append((node2_id, target_col.item(node_index)))
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
        # TODO: set seed or not? should it be deterministic?
        # Choose randomly k (num_edges) nodes to make connections with
        target_nodes = np.random.choice(ranking, size=num_edges,
                                        replace=False, p=probability)
        # Add edges to new graph
        # creating verticles while adding edges
        for target_node in target_nodes:
            new_graph.add_edge(node1_id, target_node)

    return new_graph


def recreate_by_priority_rank_random_rankings(graph):
    num_edges = round(graph.number_of_edges() / graph.number_of_nodes())
    num_of_nodes = graph.number_of_nodes()

    new_graph = nx.DiGraph()

    harmonic_number = sum([
        1 / k for k in range(1, num_of_nodes + 1)
    ])
    for node1 in graph.nodes:
        random.seed(93)
        ranking = [x for x in graph.nodes]
        random.shuffle(ranking)
        probability = [
            1 / (harmonic_number * index)
            for index, _ in enumerate(ranking, start=1)
        ]
        # TODO: set seet or not? should it be deterministic?
        target_nodes = np.random.choice(ranking, size=num_edges,
                                        replace=False, p=probability)
        for target_node in target_nodes:
            new_graph.add_edge(node1, target_node)

    return new_graph
