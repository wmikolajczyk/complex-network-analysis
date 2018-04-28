import networkx as nx
import numpy as np
import pandas as pd

from collections import defaultdict

from graph_utils import get_attributes
from neural_net_model import graph_to_training_set


# TODO: refactor to workon on datasets not graphs?
def generate_by_priority_rank(model, graph, num_edges):
    # Generate new graph
    num_of_nodes = graph.number_of_nodes()
    new_graph = nx.empty_graph(n=num_of_nodes)

    # Dict containing node ranking list for each node
    node_similarities = defaultdict(list)

    # for u in graph.nodes:
    #     attrs1 = get_attributes(graph.nodes[u].items(), 'node1_')
    #     for v in graph.nodes:
    #         attrs2 = get_attributes(graph.nodes[v].items(), 'node2_')
    #         # Dict with node1 and node2 attributes with node prefix
    #         d = {}
    #         d.update(attrs1)
    #         d.update(attrs2)
    #         # Node attributes to DataFrame
    #         feature_values = pd.DataFrame([d], columns=d.keys())
    #         # Probability of connection between nodes based on their attributes
    #         node_similarities[u].append(
    #             (v, model.predict(feature_values)[0][0]))

    df = graph_to_training_set(graph, for_recreate=True)
    x = df.shape[0] / num_of_nodes
    for i, row in enumerate(df.iterrows()):
        # int(1.1) -> 1
        u = int(i / x)
        v = int(i % x)
        node_similarities[u].append(
            (v, model.predict(np.array([row[1]]))[0][0]))

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
