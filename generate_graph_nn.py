import networkx as nx
import numpy as np
import pandas as pd

from collections import defaultdict

from keras.models import Sequential
from keras.layers import Dense


# FUNCTIONS
def attach_attributes(graph):
    degree_centralities = nx.degree_centrality(graph)
    betweenness_centralities = nx.betweenness_centrality(graph)
    closeness_centralities = nx.closeness_centrality(graph)
    pageranks = nx.pagerank(graph)

    for node_id in graph.nodes:
        node_attributes = {
            'degree_centrality': degree_centralities[node_id],
            'betweenness_centrality': betweenness_centralities[node_id],
            'closeness_centrality': closeness_centralities[node_id],
            'pagerank': pageranks[node_id]
        }
        graph.node[node_id].update(node_attributes)


def get_attributes(node_attributes, prefix):
    attributes_dict = {
        prefix + key: value
        for key, value in node_attributes
    }
    return attributes_dict


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


def generate_graph_by_nn(model, graph, num_edges):
    # Generate new graph
    new_graph = nx.empty_graph(n=graph.number_of_nodes())

    # Dict containing node ranking list for each node
    node_similarities = defaultdict(list)

    for u in graph.nodes:
        attrs1 = get_attributes(graph.nodes[u].items(), 'node1_')
        for v in graph.nodes:
            attrs2 = get_attributes(graph.nodes[v].items(), 'node2_')
            # Dict with node1 and node2 attributes with node prefix
            d = {}
            d.update(attrs1)
            d.update(attrs2)
            # Node attributes to DataFrame
            feature_values = pd.DataFrame([d], columns=d.keys())
            # Probability of connection between nodes based on their attributes
            node_similarities[u].append(
                (v, model.predict(feature_values)[0][0]))

    harmonic_number = sum([1 / k for k in range(1, graph.number_of_nodes() + 1)])

    for u in graph.nodes:
        # Node ranking sorted in descending similarity order
        ranking = [n for (n, sim) in
                   sorted(node_similarities[u], key=lambda x: x[1],
                          reverse=True)]
        # Probability of connection to node on each position at the ranking
        probability = [1 / (harmonic_number * idx)
                       for idx, elem in enumerate(ranking, start=1)]
        # Choose randomly k (num_edges) nodes to make connections with
        target_nodes = np.random.choice(ranking, size=num_edges, replace=False,
                                        p=probability)

        # Add edges to new graph
        for tn in target_nodes:
            new_graph.add_edge(u, tn)

    return new_graph
