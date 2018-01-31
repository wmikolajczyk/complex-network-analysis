# -*- coding: utf-8
import csv
import pandas as pd
import numpy as np
import networkx as nx

from keras.models import Sequential
from keras.layers import Dense

from utils import PrimarySchoolDatasetHandler, WorkplaceDatasetHandler
from config import primaryschool, workplace

# Primary School Dataset   

# Export edges to csv
PrimarySchoolDatasetHandler.export_edges(primaryschool['dataset'], primaryschool['edges'])

# Filter lower triangle from adjacency matrix
graph = nx.read_edgelist(primaryschool['edges'], create_using=nx.MultiGraph(), nodetype=int)
adj_matrix = nx.adjacency_matrix(graph)
adj_matrix_tril = np.tril(adj_matrix.todense())

# Create list of node ids
nodes_list = [x for x in graph.nodes.keys()]
# Read metadata
node_attributes = PrimarySchoolDatasetHandler.read_metadata(primaryschool['metadata'], ['class', 'gender'])

# Add attributes to the nodes
for node_id in nodes_list:
    graph.node[node_id].update(node_attributes[node_id])

#  Prepare csv with node attributes and number of connections between nodes
PrimarySchoolDatasetHandler.export_node_connections_attributes(
    nodes_list, node_attributes, adj_matrix_tril, primaryschool['prepared_dataset'])

primaryschool_df = pd.read_csv(primaryschool['prepared_dataset'], sep='\t')

# Priority Rank
new_graph = nx.MultiGraph()
num_of_edges = 3
len_of_ranking = 5

df = primaryschool_df
# TODO: fix gender Unknown values

node_number = 0
# Iterate over existing nodes
for node in nodes_list:
    # Extract node attributes
    attributes = node_attributes[node]
    node_number += 1
    # Get base node number for creating edge
    base_node_number = node_number
    # Add base node to the new graph
    new_graph.add_node(node_number, **attributes)
    # Compute ranking based on vertex attributes
    ranking = df[(df['class1']==attributes['class']) & (df['gender1']==attributes['gender'])]\
        .groupby(['class1', 'gender1', 'class2', 'gender2'], as_index=False)[['num_of_connections']]\
        .sum()\
        .sort_values(['num_of_connections'], ascending=False)\
        .head(len_of_ranking)
    # Add k number of edges
    for k in range(0, num_of_edges):
        # Sample vertex t from the ranking
        ranking_idx = np.random.choice(len(ranking))
        # Get new node attributes dict
        new_attributes = ranking.iloc[[ranking_idx]][['class2', 'gender2']]\
            .rename(columns={'class2': 'class', 'gender2': 'gender'})\
            .to_dict(orient='records')[0]
        node_number += 1
        # Add new node to the new graph
        new_graph.add_node(node_number, **new_attributes)
        # Add edge between base and new node
        new_graph.add_edge(base_node_number, node_number)


# Experimental simple model
# Clean data, add dummy columns
primaryschool_df = PrimarySchoolDatasetHandler.clean_data(primaryschool_df)
# Split dataset to X, Y
dataset = primaryschool_df.values
X = dataset[:, 0:24]
Y = dataset[:, 24]

# Create simple model
seed = 1
np.random.seed(seed)
model = Sequential()
model.add(Dense(output_dim=24, input_dim=24, activation='relu'))
model.add(Dense(output_dim=12, activation='relu'))
model.add(Dense(output_dim=1))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X, Y, epochs=10, batch_size=10000)

# Evaluate model
scores = model.evaluate(X, Y)
print('{}: {}'.format(model.metrics_names[1], scores[1]))


# Load Workplace dataset
# Read metadata
department = WorkplaceDatasetHandler.read_metadata(workplace['metadata'], ['department'])

# Prepare csv for dataframe
WorkplaceDatasetHandler.prepare_training_dataset(
    workplace['dataset'], workplace['prepared_dataset'], department)
workplace_df = pd.read_csv(workplace['prepared_dataset'], sep='\t')
