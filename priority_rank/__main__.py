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

# Prepare csv for dataframe
# TODO: update and fix
# PrimarySchoolDatasetHandler.prepare_training_dataset(
#     primaryschool['dataset'], primaryschool['prepared_data'], gender)
primaryschool_df = pd.read_csv(primaryschool['prepared_data'], sep='\t')

# Load Workplace dataset
# Read metadata
department = WorkplaceDatasetHandler.read_metadata(workplace['metadata'])

# Prepare csv for dataframe
WorkplaceDatasetHandler.prepare_training_dataset(
    workplace['dataset'], workplace['prepared_data'], department)
workplace_df = pd.read_csv(workplace['prepared_data'], sep='\t')


# Transform network to training dataset
# Primary school
primaryschool_df['class1'] = primaryschool_df['class1'].astype('category')
primaryschool_df['gender1'] = primaryschool_df['gender1'].astype('category')
primaryschool_df['class2'] = primaryschool_df['class2'].astype('category')
primaryschool_df['gender2'] = primaryschool_df['gender2'].astype('category')

cat_columns = primaryschool_df.select_dtypes(['category']).columns
primaryschool_df[cat_columns] = primaryschool_df[cat_columns].apply(lambda x: x.cat.codes)

X = primaryschool_df.iloc[:, 0:2]
Y = primaryschool_df.iloc[:, 2:]


# Very very experimental model
np.random.seed(1)
model = Sequential()
model.add(Dense(2, input_dim=2, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

model.fit(X, Y, epochs=10, batch_size=100)

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
