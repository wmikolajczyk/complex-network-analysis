import os
import csv
import shutil

import networkx as nx
import pandas as pd

from konect_graphs import file_lines

delimiter = '\t'

raw_datasets_path = 'raw_datasets'
prepared_datasets_path = 'prepared_datasets'


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

    prepared_primary_school = os.path.join(prepared_datasets_path, dataset_name)
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

    prepared_workplace = os.path.join(prepared_datasets_path, dataset_name)
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


def prepare_highschool(dataset_name, edge_list_filename, node_attributes_filename):
    raw_highschool = os.path.join(raw_datasets_path, dataset_name)
    edge_list = os.path.join(raw_highschool, edge_list_filename)
    node_attributes = os.path.join(raw_highschool, node_attributes_filename)

    prepared_highschool = os.path.join(prepared_datasets_path, dataset_name)
    prepared_edge_list = os.path.join(prepared_highschool, 'edge_list.csv')
    prepared_node_attributes = os.path.join(prepared_highschool, 'node_attributes.csv')

    if not os.path.exists(prepared_highschool):
        os.mkdir(prepared_highschool)

    #           PROCESS EDGES
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


def prepare_hospital(dataset_name):
    raw_dataset_dir = os.path.join(raw_datasets_path, dataset_name)
    edge_list_with_attributes = os.path.join(raw_dataset_dir, 'detailed_list_of_contacts_Hospital.dat')

    prepared_dataset = os.path.join(prepared_datasets_path, dataset_name)
    prepared_edge_list = os.path.join(prepared_dataset, 'edge_list.csv')
    prepared_node_attributes = os.path.join(prepared_dataset, 'node_attributes.csv')

    if not os.path.exists(prepared_dataset):
        os.mkdir(prepared_dataset)

    # PROCESS EDGES
    with open(edge_list_with_attributes, 'r') as source:
        reader = csv.reader(source, delimiter='\t')
        with open(prepared_edge_list, 'w') as result:
            writer = csv.writer(result, delimiter=delimiter)
            # GET ATTRIBUTES FOR EACH NODE
            attrs_dict = {}
            for row in reader:
                writer.writerow((row[1], row[2]))
                for node, attr in [(row[1], row[3]), (row[2], row[4])]:
                    # check if there is no different values for the same node ids
                    if node in attrs_dict:
                        if attrs_dict[node] != attr:
                            raise ValueError('Different attr values for node')
                    else:
                        attrs_dict[node] = attr

    # Convert multiple edges to weights
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

    # PROCESS ATTRIBUTES
    with open(prepared_node_attributes, 'w') as result:
        writer = csv.writer(result, delimiter=delimiter)
        writer.writerow(('node_id', 'class'))
        for key, val in attrs_dict.items():
            writer.writerow((key, val))


def prepare_moreno_blogs(dataset_name, edge_list_filename, node_attributes_filename):
    # zakladamy ze sciagniety wczesniej do raw datasets
    raw_dataset_dir = os.path.join(raw_datasets_path, dataset_name)
    edge_list = os.path.join(raw_dataset_dir, edge_list_filename)
    node_attributes = os.path.join(raw_dataset_dir, node_attributes_filename)

    prepared_dataset = os.path.join(prepared_datasets_path, dataset_name)
    prepared_edge_list = os.path.join(prepared_dataset, 'edge_list.csv')
    prepared_node_attributes = os.path.join(prepared_dataset, 'node_attributes.csv')

    if not os.path.exists(prepared_dataset):
        os.mkdir(prepared_dataset)

    # PROCESS EDGES
    # There is nice edge list - need to be converted to weights only
    directed_graph = nx.read_edgelist(edge_list, create_using=nx.DiGraph(), comments='%', nodetype=int)
    # graph is already directed
    # set all weights equal to 1
    nx.set_edge_attributes(directed_graph, name='weight', values=1)
    nx.write_edgelist(directed_graph, prepared_edge_list, delimiter=delimiter)

    # PROCESS ATTRIBUTES
    sorted_nodes = sorted([int(x) for x in directed_graph.nodes])
    # sprawdz czy liczba wierszy jest taka jak liczba wierzcholkow

    if not file_lines(node_attributes) == len(sorted_nodes):
        raise ValueError('Number of nodes and number of lines in attributes file are not the same')
    # idz po kolei po id wierzcholkow i im tworz cechy
    with open(node_attributes, 'r') as source:
        # actually it's only one value so delimiter has no effect
        reader = csv.reader(source)
        with open(prepared_node_attributes, 'w') as result:
            writer = csv.writer(result, delimiter=delimiter)
            writer.writerow(('node_id', 'orientation'))
            # attrs - list, but here it's only one element
            for node_id, attrs in zip(sorted_nodes, reader):
                writer.writerow((node_id, attrs[0]))


def prepare_moreno_sheep(dataset_name, edge_list_filename, node_attributes_filename):
    raw_dataset_dir = os.path.join(raw_datasets_path, dataset_name)
    edge_list = os.path.join(raw_dataset_dir, edge_list_filename)
    node_attributes = os.path.join(raw_dataset_dir, node_attributes_filename)

    prepared_dataset = os.path.join(prepared_datasets_path, dataset_name)
    prepared_edge_list = os.path.join(prepared_dataset, 'edge_list.csv')
    prepared_node_attributes = os.path.join(prepared_dataset, 'node_attributes.csv')

    if not os.path.exists(prepared_dataset):
        os.mkdir(prepared_dataset)
    # Graph id Directed and have not Multi edges
    #   so there is no need to sum weights
    directed_weighted_graph = nx.read_edgelist(edge_list, create_using=nx.DiGraph(), comments='%', nodetype=int, data=(('weight', float),))
    nx.write_edgelist(directed_weighted_graph, prepared_edge_list, delimiter=delimiter)

    # PROCESS ATTRIBUTES
    sorted_nodes = sorted(directed_weighted_graph.nodes)

    if not file_lines(node_attributes) == len(sorted_nodes):
        raise ValueError('Number of nodes and number of lines in attributes file are not the same')

    with open(node_attributes, 'r') as source:
        reader = csv.reader(source)
        with open(prepared_node_attributes, 'w') as result:
            writer = csv.writer(result, delimiter=delimiter)
            writer.writerow(('node_id', 'age'))
            for node_id, attrs in zip(sorted_nodes, reader):
                try:
                    writer.writerow((node_id, attrs[0]))
                except IndexError:
                    writer.writerow((node_id, ''))

# prepare_primary_school('primary_school')
# prepare_workplace('workplace')
# TODO: refactor - load primary school like highschool - maybe load workplace too
# prepare_highschool('highschool_2011', 'thiers_2011.csv', 'metadata_2011.txt')
# prepare_highschool('highschool_2012', 'thiers_2012.csv', 'metadata_2012.txt')
# prepare_hospital('hospital')
# prepare_moreno_blogs('moreno_blogs', 'out.moreno_blogs_blogs', 'ent.moreno_blogs_blogs.blog.orientation')
# prepare_moreno_sheep('moreno_sheep', 'out.moreno_sheep_sheep', 'ent.moreno_sheep_sheep.sheep.age')
print('done')






def prepare_households(dataset_name):
    # NOT WORKING - LIKELY TO DELETE
    # id is summed from house_letter + id
    raw_households = os.path.join(raw_datasets_path, dataset_name)
    edge_list_with_attributes = os.path.join(raw_households, 'scc2034_kilifi_all_contacts_within_households.csv')

    prepared_households = os.path.join(prepared_datasets_path, dataset_name)
    prepared_edge_list = os.path.join(prepared_households, 'edge_list.csv')
    prepared_node_attributes = os.path.join(prepared_households, 'node_attributes.csv')

    if not os.path.exists(prepared_households):
        os.mkdir(prepared_households)

    #           PROCESS EDGES
    with open(edge_list_with_attributes, 'r') as source:
        reader = csv.reader(source, delimiter=',')
        with open(prepared_edge_list, 'w') as result:
            writer = csv.writer(result, delimiter=delimiter)
            # skip first row
            next(reader)
            for row in reader:
                writer.writerow((row[1], row[3], {'weight': int(row[8])}))
    # multigraph - duration as weight
    weighted_graph = nx.Graph()
    multi_graph = nx.read_edgelist(prepared_edge_list, create_using=nx.MultiGraph())

    for u, v, data in multi_graph.edges(data=True):
        if weighted_graph.has_edge(u, v):
            weighted_graph[u][v]['weight'] += data['weight']
        else:
            weighted_graph.add_edge(u, v, weight=data['weight'])
    # Convert graph to directed
    directed_weighted_graph = weighted_graph.to_directed()
    nx.write_edgelist(directed_weighted_graph, prepared_edge_list, delimiter=delimiter)
    # PROCESS ATTRIBUTES
    edge_attrs_df = pd.read_csv(edge_list_with_attributes, delimiter=',')
    edge_attrs_df = edge_attrs_df.drop(columns=['duration', 'day', 'hour'])
    edge_attrs_df = edge_attrs_df.drop_duplicates()
    if not sorted(edge_attrs_df['m1'].unique()) == sorted(edge_attrs_df['m2'].unique()):
        raise ValueError('m1 id numbers are not equal to m2 id numbeedge_attrs_dfrs')
    #unique_node_attrs = edge_attrs_df.groupby(['m1'])
    attrs_df = edge_attrs_df[['h1', 'm1', 'age1', 'sex1']]
    import pdb; pdb.set_trace()
