import os
import csv
import shutil
import json

import networkx as nx
# podaj sciezke do folderu z datasetem
# przetworz
# podaj sciezke docelowa - zwroc folder - edges - attributes

# GOAL: Create directory with 3 files - edge_list, node_attribute, graph_meta

delimiter = '\t'

# EVERY GRAPH TREAT AS WEIGHTED AND DIRECTED
def prepare_primary_school(dataset_dir):
    """
    Primary school temporal network data (2 features)
        features: class, gender
        link: http://www.sociopatterns.org/datasets/primary-school-temporal-network-data/

    Contacts between children and teachers in the study published in BMC Infectious Diseases 2014
    Edge = contact between people

    Timestamps of contact -> to remove, but count how many connections and set it as edge weights
    """
    edge_list = os.path.join(dataset_dir, 'primaryschool.csv')
    node_attributes = os.path.join(dataset_dir, 'metadata_primaryschool.txt')

    prepared_primary_school = 'prepared_datasets/primary_school'
    prepared_edge_list = os.path.join(prepared_primary_school, 'edge_list.csv')
    prepared_node_attributes = os.path.join(prepared_primary_school, 'node_attributes.csv')
    prepared_meta = os.path.join(prepared_primary_school, 'meta.json')

    if not os.path.exists(prepared_primary_school):
        os.mkdir(prepared_primary_school)

    #           PROCESS EDGES
    # 2nd and 3rd column contains node number - 4th and 5th are attribute duplicated from meta file
    # 31220 1558    1567    3B  3B
    with open(edge_list, 'r') as source:
        reader = csv.reader(source, delimiter=delimiter)
        with open(prepared_edge_list, 'w') as result:
            writer = csv.writer(result, delimiter=delimiter)
            for row in reader:
                writer.writerow((row[1], row[2]))

    #               Convert multiple edges to weights
    weighted_graph = nx.Graph()
    multi_graph = nx.read_edgelist(prepared_edge_list, create_using=nx.MultiGraph())

    for u, v, data in multi_graph.edges(data=True):
        if weighted_graph.has_edge(u, v):
            weighted_graph[u][v]['weight'] += 1
        else:
            weighted_graph.add_edge(u, v, weight=1)
    # GET DIRECTED GRAPH (edge a->b {weight: 2} = a->b {weight: 2} and b->a {weight: 2})
    directed_weighted_graph = weighted_graph.to_directed()
    nx.write_edgelist(directed_weighted_graph, prepared_edge_list, delimiter=delimiter)

    #           PROCESS ATTRIBUTES
    shutil.copy(node_attributes, prepared_node_attributes)


prepare_primary_school('raw_datasets/primary_school')
print('done')