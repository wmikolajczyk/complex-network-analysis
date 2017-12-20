# -*- coding: utf-8
import csv
import pandas as pd
import numpy as np
import networkx as nx

from keras.models import Sequential
from keras.layers import Dense

from utils import PrimarySchoolDatasetHandler, WorkplaceDatasetHandler
from config import primaryschool, workplace

# Load Primary School dataset
# Read metadata
# zbierz atrybuty wierzchołków
node_attributes = PrimarySchoolDatasetHandler.read_metadata(primaryschool['metadata'])

# weź listę wierzchołków z metadata
graph = nx.read_edgelist(primaryschool['prepared_graph_dataset'], create_using=nx.MultiGraph(), nodetype=int)
A = nx.adjacency_matrix(graph)
lowerA = np.tril(A.todense())
nodes_list = [x for x in graph.nodes.keys()]

# Create csv with node ids and num of edges between nodes
# it gives an output with 29161 rows which is ok because
# 242^2 = 29282
# 29282 - 29161 = 121
# 121 is 242 (number of nodes) / 2
# because we take lower triangle of matrix
with open('Datasets/primary_school/prepared/node_connections.csv', 'w') as result:
    writer = csv.writer(result, delimiter='\t')
    for i in range(1, len(nodes_list)):
        node1_id = nodes_list[i]
        for j in range(i):
            node2_id = nodes_list[j]
            num_of_edges = lowerA[i][j]
            writer.writerow((node1_id, node2_id, num_of_edges))

# Create csv with node attributes and num of edges between nodes
with open('Datasets/primary_school/prepared/node_connections_attributes.csv', 'w') as result:
    writer = csv.writer(result, delimiter='\t')
    writer.writerow(
        ('class1', 'gender1', 'class2', 'gender2', 'num_of_connections')
    )
    for i in range(1, len(nodes_list)):
        node1_id = nodes_list[i]
        node1_attrs = list(node_attributes[node1_id].values())
        for j in range(i):
            node2_id = nodes_list[j]
            node2_attrs = list(node_attributes[node2_id].values())
            num_of_edges = lowerA[i][j]
            writer.writerow((
                node1_attrs[0], node1_attrs[1],
                node2_attrs[0], node2_attrs[1],
                num_of_edges
            ))

# Prepare csv for dataframe
# TODO: update and fix
# PrimarySchoolDatasetHandler.prepare_training_dataset(
#     primaryschool['dataset'], primaryschool['prepared_data'], gender)
primaryschool_df = pd.read_csv(primaryschool['prepared_data'], sep='\t')


# Transform network to training dataset
# Primary school
# Remove 'Unknown' values in gender columns
# gender1
female1 = primaryschool_df['gender1'].value_counts()['F']
female1_prob = female1 / primaryschool_df.shape[0]
gender_to_replace = ['M', 'F'][female1_prob >= 0.5]
primaryschool_df['gender1'] = primaryschool_df['gender1'].replace('Unknown', gender_to_replace)

# gender2
female2 = primaryschool_df['gender2'].value_counts()['F']
female2_prob = female2 / primaryschool_df.shape[0]
gender_to_replace = ['M', 'F'][female2_prob >= 0.5]
primaryschool_df['gender2'] = primaryschool_df['gender2'].replace('Unknown', gender_to_replace)

primaryschool_df['gender1'] = primaryschool_df['gender1'].astype('category')
primaryschool_df['gender2'] = primaryschool_df['gender2'].astype('category')

cat_columns = primaryschool_df.select_dtypes(['category']).columns
primaryschool_df[cat_columns] = primaryschool_df[cat_columns].apply(lambda x: x.cat.codes)

primaryschool_df = pd.get_dummies(primaryschool_df, columns=['class1', 'class2'])

cols = primaryschool_df.columns.tolist()
# move num_of_connections to the last index
cols[2], cols[len(cols) - 1] = cols[len(cols) - 1], cols[2]

primaryschool_df = primaryschool_df[cols]

X = primaryschool_df.iloc[:, 0:24]
Y = primaryschool_df.iloc[:, 24]


# Very very experimental model
np.random.seed(1)
model = Sequential()
model.add(Dense(output_dim=36, input_dim=24, activation='relu'))
model.add(Dense(output_dim=1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

model.fit(X, Y, epochs=10)

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
    workplace['dataset'], workplace['prepared_data'], department)
workplace_df = pd.read_csv(workplace['prepared_data'], sep='\t')