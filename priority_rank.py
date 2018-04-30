import networkx as nx
import numpy as np

from collections import defaultdict

from neural_net_model import graph_to_dataframe


# TODO: refactor to workon on datasets not graphs?
def generate_by_priority_rank(model, graph, num_edges):
    # Generate new graph
    num_of_nodes = graph.number_of_nodes()
    new_graph = nx.empty_graph(n=num_of_nodes)

    # Dict containing node ranking list for each node
    node_similarities = defaultdict(list)

    df = graph_to_dataframe(graph, remove_target_col=True)
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
