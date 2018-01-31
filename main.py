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
node_attributes = PrimarySchoolDatasetHandler.read_metadata(primaryschool['metadata'], ['class', 'gender'])
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

primaryschool_df = PrimarySchoolDatasetHandler.clean_data(primaryschool_df)

# TODO: export primaryschool_df to csv and read it from csv
# primaryschool_df.to_csv('prepared_dataset.csv', index=False)

#   - Split dataset to X, Y
dataset = primaryschool_df.values
X = dataset[:, 0:24]
Y = dataset[:, 24]

#   - Create simple model
seed = 1
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

# TODO: Priority Rank
"""
df = pd.read_csv(primaryschool['prepared_dataset'], sep='\t')
df['gender1'] = df['gender1'].replace('Unknown', 'M')
df['gender2'] = df['gender2'].replace('Unknown', 'M')
df.groupby(['class1', 'gender1', 'class2', 'gender2']).sum()
df[(df['class1']=='1A') & (df['gender1']=='M')].groupby(['class1', 'gender1', 'class2', 'gender2']).sum()
# 1A and M are placeholder values - to do - extract from node
# get list of dicts / rows in dataframe class1, gender1, num_of_connections - sorted
df[(df['class1']=='1A') & (df['gender1']=='M')].groupby(['class1', 'gender1', 'class2', 'gender2'])[['num_of_connections']].sum()
df[(df['class1']=='1A') & (df['gender1']=='M')].groupby(['class1', 'gender1', 'class2', 'gender2'], as_index=False)[['num_of_connections']].sum().sort_values(['num_of_connections'], ascending=False)
"""

# add attributes to the graph
for node_id in nodes_list:
    for attribute in node_attributes[node_id].items():
        graph.node[node_id][attribute[0]] = attribute[1]
       
new_graph = nx.MultiGraph()
num_of_edges = 3
len_of_ranking = 5

df = pd.read_csv(primaryschool['prepared_dataset'], sep='\t')
# df['gender1'] = df['gender1'].replace('Unknown', 'M')
# df['gender2'] = df['gender2'].replace('Unknown', 'M')

node_number = 0
for node in nodes_list:
    attributes = node_attributes[node]
    node_number += 1
    base_node_number = node_number
    new_graph.add_node(node_number, **attributes)
    # compute ranking (len_of_ranking - length) based on vertex attributes
    # predict attributes of the vertex to connect
    
    ranking = df[(df['class1']==attributes['class']) & (df['gender1']==attributes['gender'])]\
        .groupby(['class1', 'gender1', 'class2', 'gender2'], as_index=False)[['num_of_connections']]\
        .sum()\
        .sort_values(['num_of_connections'], ascending=False)\
        .head(len_of_ranking)
    for i in range(0, num_of_edges):
        # sample vertex t from the ranking
        ranking_idx = np.random.choice(len(ranking))
        # graph add edge node - ranking[ranking_idx]
        new_attributes = ranking.iloc[[ranking_idx]][['class2', 'gender2']]\
            .rename(columns={'class2': 'class', 'gender2': 'gender'})\
            .to_dict(orient='records')[0]
        node_number += 1
        new_graph.add_node(node_number, **new_attributes)
        new_graph.add_edge(base_node_number, node_number)

# Load Workplace dataset
# Read metadata
department = WorkplaceDatasetHandler.read_metadata(workplace['metadata'], ['department'])

# Prepare csv for dataframe
WorkplaceDatasetHandler.prepare_training_dataset(
    workplace['dataset'], workplace['prepared_dataset'], department)
workplace_df = pd.read_csv(workplace['prepared_dataset'], sep='\t')
