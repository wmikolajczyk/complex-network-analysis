import pandas as pd
import networkx as nx
import numpy as np

from config import primaryschool
from utils import PrimarySchoolDatasetHandler

def priority_rank(df, nodes_list, node_attributes):
# Priority Rank
    new_graph = nx.MultiGraph()
    num_of_edges = 3
    len_of_ranking = 5

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
    return new_graph

if __name__ == '__main__':
	# TODO: fix gender Unknown values
	primaryschool_df = pd.read_csv(primaryschool['prepared_dataset'], sep='\t')
	# Read graph
	graph = nx.read_edgelist(primaryschool['edges'], create_using=nx.MultiGraph(), nodetype=int)
	# Create list of node ids
	nodes_list = [x for x in graph.nodes.keys()]
    # Read metadata
	node_attributes = PrimarySchoolDatasetHandler.read_metadata(primaryschool['metadata'], ['class', 'gender'])

    # Add attributes to the nodes
	for node_id in nodes_list:
	    graph.node[node_id].update(node_attributes[node_id])

	priority_rank_graph = priority_rank(primaryschool_df, nodes_list, node_attributes)
	print('Done')
