import networkx as nx
import numpy as np
import pandas as pd

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

# Create graph, attach attributes -> to DataFrame
n1, p1 = 10, 0.8

graph1 = nx.gnp_random_graph(n1, p1, seed=93)
attach_attributes(graph1)

graph1_data = graph_to_training_set(graph1)
df = pd.DataFrame(graph1_data)

# Split DF into X and y
X_train = df.iloc[:, :8]
y_train = df.iloc[:, 8]

# Create model
model = Sequential()
model.add(Dense(units=8, input_dim=8, activation='sigmoid'))
model.add(Dense(units=1))

model.compile(loss='binary_crossentropy', optimizer='sgd')

# Train model
model.fit(X_train, y_train, epochs=100)

# Generate new graph
new_graph = nx.empty_graph(n=graph1.number_of_nodes())

num_edges = 3

node_similarities = {}

for u in graph1.nodes:
    node_similarities[u] = []

    for v in graph1.nodes:
        d = {}
        d.update(get_attributes(graph1.nodes[u].items(), 'node1_'))
        d.update(get_attributes(graph1.nodes[v].items(), 'node2_'))

        feature_values = pd.DataFrame([d], columns=d.keys())

        node_similarities[u].append((v, model.predict(feature_values)[0][0]))

h_n = sum([1/k for k in range(1, graph1.number_of_nodes() + 1)])

for u in graph1.nodes:
    ranking = [n for (n, sim) in sorted(node_similarities[u], key=lambda x: x[1], reverse=True)]
    prob = [1/(h_n * idx) for idx, elem in enumerate(ranking, start=1)]

    target_nodes = np.random.choice(ranking, size=num_edges, replace=False, p=prob)

    for tn in target_nodes:
        new_graph.add_edge(u, tn)

print(nx.adj_matrix(new_graph).todense())
