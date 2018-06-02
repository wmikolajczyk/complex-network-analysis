import os
import json

import networkx as nx
import pandas as pd


def load_primary_school():
    prepared_primary_school = 'prepared_datasets/primary_school'
    prepared_edge_list = os.path.join(prepared_primary_school, 'edge_list.csv')
    prepared_node_attributes = os.path.join(prepared_primary_school, 'node_attributes.csv')
    prepared_meta = os.path.join(prepared_primary_school, 'meta.json')

    with open(prepared_meta, 'r') as src:
        meta = json.load(src)

    #       LOAD EDGES
    graph = nx.read_edgelist(prepared_edge_list, nodetype=int)
    # print(graph.edges(data=True))
    #       ADD ATTRIBUTES
    """
    iteruj po wierzcholkach
    znajdz wiersz w pliku z atrybutami
        przeczytaj atrybuty
        dodaj atrybuty
    """
    attributes_data = pd.read_csv(prepared_node_attributes, delimiter='\t', header=None)
    attributes_data.columns = ['node_id', 'class', 'gender']
    attributes_columns = list(attributes_data.columns)
    attributes_columns.remove('node_id')
    for node_id in graph.nodes:
        attrs = attributes_data.loc[attributes_data['node_id'] == node_id]
        node_attributes = {
            colname: attrs[colname].values[0]
            for colname in attributes_columns
        }
        graph.node[node_id].update(node_attributes)
    return graph

g = load_primary_school()
print('done')
