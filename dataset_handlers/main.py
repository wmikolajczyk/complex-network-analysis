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
