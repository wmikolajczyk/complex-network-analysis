# -*- coding: utf-8
import csv
import pandas as pd
import numpy as np
import networkx as nx

from keras.models import Sequential
from keras.layers import Dense

from utils import PrimarySchoolDatasetHandler, WorkplaceDatasetHandler
from config import primaryschool, workplace

# Primary School
#   - Metadata
node_attributes = PrimarySchoolDatasetHandler.read_metadata(primaryschool['metadata'])
#   - Export edges to csv
PrimarySchoolDatasetHandler.export_edges(primaryschool['dataset'], primaryschool['edges'])
#   - Filter lower triangle from adjacency matrix
graph = nx.read_edgelist(primaryschool['edges'], create_using=nx.MultiGraph(), nodetype=int)
adj_matrix = nx.adjacency_matrix(graph)
adj_matrix_tril = np.tril(adj_matrix.todense())

nodes_list = [x for x in graph.nodes.keys()]

#   - Prepare csv with node attributes and number of connections between nodes
PrimarySchoolDatasetHandler.export_node_connections_attributes(
    nodes_list, node_attributes, adj_matrix_tril, primaryschool['prepared_dataset'])

primaryschool_df = pd.read_csv(primaryschool['prepared_dataset'], sep='\t')

#   - Transform data to training dataset
#       remove 'Unknown' values in gender columns
female1 = primaryschool_df['gender1'].value_counts()['F']
female1_prob = female1 / primaryschool_df.shape[0]
gender_to_replace = ['M', 'F'][female1_prob >= 0.5]
primaryschool_df['gender1'] = primaryschool_df['gender1'].replace('Unknown', gender_to_replace)

female2 = primaryschool_df['gender2'].value_counts()['F']
female2_prob = female2 / primaryschool_df.shape[0]
gender_to_replace = ['M', 'F'][female2_prob >= 0.5]
primaryschool_df['gender2'] = primaryschool_df['gender2'].replace('Unknown', gender_to_replace)

#   - Mark gender columns as categorical and apply encoding
primaryschool_df['gender1'] = primaryschool_df['gender1'].astype('category')
primaryschool_df['gender2'] = primaryschool_df['gender2'].astype('category')

cat_columns = primaryschool_df.select_dtypes(['category']).columns
primaryschool_df[cat_columns] = primaryschool_df[cat_columns].apply(lambda x: x.cat.codes)

#   - Create dummies from class categorical columns which have more than 2 different values
primaryschool_df = pd.get_dummies(primaryschool_df, columns=['class1', 'class2'])

#   - Change dataframe column ordering (move num_of_connections to the last index)
cols = primaryschool_df.columns.tolist()
cols[2], cols[len(cols) - 1] = cols[len(cols) - 1], cols[2]

primaryschool_df = primaryschool_df[cols]

#   - Split dataset to X, Y
dataset = primaryschool_df.values
X = dataset[:, 0:24]
Y = dataset[:, 24]

#   - Create simple model
seed=1
np.random.seed(seed)
model = Sequential()
model.add(Dense(output_dim=24, input_dim=24, activation='relu'))
model.add(Dense(output_dim=12, activation='relu'))
model.add(Dense(output_dim=1))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

#   - Train model
model.fit(X, Y, epochs=10, batch_size=10000)

#   - Evaluate model
scores = model.evaluate(X, Y)
print('{}: {}'.format(model.metrics_names[1], scores[1]))

# Priority Rank
"""
for each vertex n
    compute ranking R (m-lenght)
    for number of edges for vertex
        sample vertex t from the ranking
        add edge vertex - vertex t
"""

# Load Workplace dataset
# Read metadata
department = WorkplaceDatasetHandler.read_metadata(workplace['metadata'])

# Prepare csv for dataframe
WorkplaceDatasetHandler.prepare_training_dataset(
    workplace['dataset'], workplace['prepared_dataset'], department)
workplace_df = pd.read_csv(workplace['prepared_dataset'], sep='\t')