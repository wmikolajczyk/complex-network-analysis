import os
import csv

import networkx as nx
import pandas as pd


delimiter = '\t'

raw_datasets_path = 'raw_datasets'
prepared_datsets_path = 'prepared_datasets'


# EVERY GRAPH TREAT AS WEIGHTED AND DIRECTED
def prepare_primary_school(dataset_name):
    """
    Primary school temporal network data (2 features)
        features: class, gender
        link: http://www.sociopatterns.org/datasets/primary-school-temporal-network-data/

    Contacts between children and teachers in the study published in BMC Infectious Diseases 2014
    Edge = contact between people

    Timestamps of contact -> to remove, but count how many connections and set it as edge weights
    """
    raw_primary_school = os.path.join(raw_datasets_path, dataset_name)
    edge_list = os.path.join(raw_primary_school, 'primaryschool.csv')
    node_attributes = os.path.join(raw_primary_school, 'metadata_primaryschool.txt')

    prepared_primary_school = os.path.join(prepared_datsets_path, dataset_name)
    prepared_edge_list = os.path.join(prepared_primary_school, 'edge_list.csv')
    prepared_node_attributes = os.path.join(prepared_primary_school, 'node_attributes.csv')

    if not os.path.exists(prepared_primary_school):
        os.mkdir(prepared_primary_school)

    #           PROCESS EDGES
    # 2nd and 3rd column contains node number - 4th and 5th are attribute duplicated from meta file
    # 31220 1558    1567    3B  3B
    with open(edge_list, 'r') as source:
        reader = csv.reader(source, delimiter='\t')
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
    attrs_df = pd.read_csv(node_attributes, delimiter=delimiter, 
        names=['node_id', 'class', 'gender'])
    attrs_df.to_csv(prepared_node_attributes, sep=delimiter, index=False)


def prepare_workplace(dataset_name):
    raw_workplace = os.path.join(raw_datasets_path, dataset_name)
    edge_list = os.path.join(raw_workplace, 'tij_InVS.dat')
    node_attributes = os.path.join(raw_workplace, 'metadata_InVS13.txt')

    prepared_workplace = os.path.join(prepared_datsets_path, dataset_name)
    prepared_edge_list = os.path.join(prepared_workplace, 'edge_list.csv')
    prepared_node_attributes = os.path.join(prepared_workplace, 'node_attributes.csv')

    if not os.path.exists(prepared_workplace):
        os.mkdir(prepared_workplace)

    #           PROCESS EDGES
    with open(edge_list, 'r') as source:
        reader = csv.reader(source, delimiter=' ')
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
    #               Convert graph to directed
    directed_weighted_graph = weighted_graph.to_directed()
    nx.write_edgelist(directed_weighted_graph, prepared_edge_list, delimiter=delimiter)
    #           PROCESS ATTRIBUTES
    attrs_df = pd.read_csv(node_attributes, delimiter='\t', names=['node_id', 'department'])
    attrs_df.to_csv(prepared_node_attributes, sep=delimiter, index=False)


# prepare_primary_school('primary_school')
prepare_workplace('workplace')
print('done')
