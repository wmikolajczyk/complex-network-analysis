import os
import csv

import networkx as nx
import pandas as pd

from konect_graphs import file_lines

delimiter = '\t'


def get_paths_dict(dataset_name, edge_list_filename, node_attributes_filename=None):
    raw_datasets_path = 'raw_datasets'
    prepared_datasets_path = 'prepared_datasets'

    raw_dataset_dir = os.path.join(raw_datasets_path, dataset_name)
    prepared_dataset_dir = os.path.join(prepared_datasets_path, dataset_name)

    path_dict = {
        'edge_list': os.path.join(raw_dataset_dir, edge_list_filename),

        'prepared_dataset_dir': prepared_dataset_dir,
        'prepared_edge_list': os.path.join(prepared_dataset_dir, 'edge_list.csv'),
        'prepared_node_attributes': os.path.join(prepared_dataset_dir, 'node_attributes.csv')
    }
    if node_attributes_filename:
        path_dict['node_attributes'] = os.path.join(raw_dataset_dir, node_attributes_filename)
    return path_dict


# EVERY GRAPH TREAT AS WEIGHTED AND DIRECTED
def prepare_primary_school(dataset_name, edge_list_filename, node_attributes_filename):
    """
    Primary school temporal network data (2 features)
        features: class, gender
        link: http://www.sociopatterns.org/datasets/primary-school-temporal-network-data/

    Contacts between children and teachers in the study published in BMC Infectious Diseases 2014
    Edge = contact between people

    Timestamps of contact -> to remove, but count how many connections and set it as edge weights
    """
    paths = get_paths_dict(dataset_name, edge_list_filename, node_attributes_filename)

    if not os.path.exists(paths['prepared_dataset_dir']):
        os.mkdir(paths['prepared_dataset_dir'])

    #           PROCESS EDGES
    # 2nd and 3rd column contains node number - 4th and 5th are attribute duplicated from meta file
    # 31220 1558    1567    3B  3B
    with open(paths['edge_list'], 'r') as source:
        reader = csv.reader(source, delimiter='\t')
        with open(paths['prepared_edge_list'], 'w') as result:
            writer = csv.writer(result, delimiter=delimiter)
            for row in reader:
                writer.writerow((row[1], row[2]))

    #               Convert multiple edges to weights
    weighted_graph = nx.Graph()
    multi_graph = nx.read_edgelist(paths['prepared_edge_list'], create_using=nx.MultiGraph())

    for u, v, data in multi_graph.edges(data=True):
        if weighted_graph.has_edge(u, v):
            weighted_graph[u][v]['weight'] += 1
        else:
            weighted_graph.add_edge(u, v, weight=1)
    # GET DIRECTED GRAPH (edge a->b {weight: 2} = a->b {weight: 2} and b->a {weight: 2})
    directed_weighted_graph = weighted_graph.to_directed()
    nx.write_edgelist(directed_weighted_graph, paths['prepared_edge_list'], delimiter=delimiter)

    #           PROCESS ATTRIBUTES
    attrs_df = pd.read_csv(paths['node_attributes'], delimiter=delimiter,
        names=['node_id', 'class', 'gender'])
    attrs_df.to_csv(paths['prepared_node_attributes'], sep=delimiter, index=False)


def prepare_workplace(dataset_name, edge_list_filename, node_attributes_filename):
    paths = get_paths_dict(dataset_name, edge_list_filename, node_attributes_filename)

    if not os.path.exists(paths['prepared_dataset_dir']):
        os.mkdir(paths['prepared_dataset_dir'])

    #           PROCESS EDGES
    with open(paths['edge_list'], 'r') as source:
        reader = csv.reader(source, delimiter=' ')
        with open(paths['prepared_edge_list'], 'w') as result:
            writer = csv.writer(result, delimiter=delimiter)
            for row in reader:
                writer.writerow((row[1], row[2]))
    #               Convert multiple edges to weights
    weighted_graph = nx.Graph()
    multi_graph = nx.read_edgelist(paths['prepared_edge_list'], create_using=nx.MultiGraph())

    for u, v, data in multi_graph.edges(data=True):
        if weighted_graph.has_edge(u, v):
            weighted_graph[u][v]['weight'] += 1
        else:
            weighted_graph.add_edge(u, v, weight=1)
    #               Convert graph to directed
    directed_weighted_graph = weighted_graph.to_directed()
    nx.write_edgelist(directed_weighted_graph, paths['prepared_edge_list'], delimiter=delimiter)
    #           PROCESS ATTRIBUTES
    attrs_df = pd.read_csv(paths['node_attributes'], delimiter='\t', names=['node_id', 'department'])
    attrs_df.to_csv(paths['prepared_node_attributes'], sep=delimiter, index=False)

#
def prepare_highschool(dataset_name, edge_list_filename, node_attributes_filename):
    paths = get_paths_dict(dataset_name, edge_list_filename, node_attributes_filename)

    if not os.path.exists(paths['prepared_dataset_dir']):
        os.mkdir(paths['prepared_dataset_dir'])

    #           PROCESS EDGES
    with open(paths['edge_list'], 'r') as source:
        reader = csv.reader(source, delimiter='\t')
        with open(paths['prepared_edge_list'], 'w') as result:
            writer = csv.writer(result, delimiter=delimiter)
            for row in reader:
                writer.writerow((row[1], row[2]))

    #               Convert multiple edges to weights
    weighted_graph = nx.Graph()
    multi_graph = nx.read_edgelist(paths['prepared_edge_list'], create_using=nx.MultiGraph())

    for u, v, data in multi_graph.edges(data=True):
        if weighted_graph.has_edge(u, v):
            weighted_graph[u][v]['weight'] += 1
        else:
            weighted_graph.add_edge(u, v, weight=1)
    # GET DIRECTED GRAPH (edge a->b {weight: 2} = a->b {weight: 2} and b->a {weight: 2})
    directed_weighted_graph = weighted_graph.to_directed()
    nx.write_edgelist(directed_weighted_graph, paths['prepared_edge_list'], delimiter=delimiter)

    #           PROCESS ATTRIBUTES
    attrs_df = pd.read_csv(paths['node_attributes'], delimiter=delimiter,
        names=['node_id', 'class', 'gender'])
    attrs_df.to_csv(paths['prepared_node_attributes'], sep=delimiter, index=False)


def prepare_hospital(dataset_name, edge_list_filename):
    paths = get_paths_dict(dataset_name, edge_list_filename)
    edge_list_with_attributes = paths['edge_list']

    if not os.path.exists(paths['prepared_dataset_dir']):
        os.mkdir(paths['prepared_dataset_dir'])

    # PROCESS EDGES
    with open(edge_list_with_attributes, 'r') as source:
        reader = csv.reader(source, delimiter='\t')
        with open(paths['prepared_edge_list'], 'w') as result:
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
    multi_graph = nx.read_edgelist(paths['prepared_edge_list'], create_using=nx.MultiGraph())

    for u, v, data in multi_graph.edges(data=True):
        if weighted_graph.has_edge(u, v):
            weighted_graph[u][v]['weight'] += 1
        else:
            weighted_graph.add_edge(u, v, weight=1)
    # GET DIRECTED GRAPH (edge a->b {weight: 2} = a->b {weight: 2} and b->a {weight: 2})
    directed_weighted_graph = weighted_graph.to_directed()
    nx.write_edgelist(directed_weighted_graph, paths['prepared_edge_list'], delimiter=delimiter)

    # PROCESS ATTRIBUTES
    with open(paths['prepared_node_attributes'], 'w') as result:
        writer = csv.writer(result, delimiter=delimiter)
        writer.writerow(('node_id', 'class'))
        for key, val in attrs_dict.items():
            writer.writerow((key, val))


def prepare_moreno_blogs(dataset_name, edge_list_filename, node_attributes_filename):
    # zakladamy ze sciagniety wczesniej do raw datasets
    paths = get_paths_dict(dataset_name, edge_list_filename, node_attributes_filename)

    if not os.path.exists(paths['prepared_dataset_dir']):
        os.mkdir(paths['prepared_dataset_dir'])

    # PROCESS EDGES
    # There is nice edge list - need to be converted to weights only
    directed_graph = nx.read_edgelist(paths['edge_list'], create_using=nx.DiGraph(), comments='%', nodetype=int)
    # graph is already directed
    # set all weights equal to 1
    nx.set_edge_attributes(directed_graph, name='weight', values=1)
    nx.write_edgelist(directed_graph, paths['prepared_edge_list'], delimiter=delimiter)

    # PROCESS ATTRIBUTES
    sorted_nodes = sorted([int(x) for x in directed_graph.nodes])
    # sprawdz czy liczba wierszy jest taka jak liczba wierzcholkow

    if not file_lines(paths['node_attributes']) == len(sorted_nodes):
        raise ValueError('Number of nodes and number of lines in attributes file are not the same')
    # idz po kolei po id wierzcholkow i im tworz cechy
    with open(paths['node_attributes'], 'r') as source:
        # actually it's only one value so delimiter has no effect
        reader = csv.reader(source)
        with open(paths['prepared_node_attributes'], 'w') as result:
            writer = csv.writer(result, delimiter=delimiter)
            writer.writerow(('node_id', 'orientation'))
            # attrs - list, but here it's only one element
            for node_id, attrs in zip(sorted_nodes, reader):
                writer.writerow((node_id, attrs[0]))


def prepare_moreno_sheep(dataset_name, edge_list_filename, node_attributes_filename):
    paths = get_paths_dict(dataset_name, edge_list_filename, node_attributes_filename)

    if not os.path.exists(paths['prepared_dataset_dir']):
        os.mkdir(paths['prepared_dataset_dir'])

    # PROCESS EDGES
    # Graph id Directed and have not Multi edges
    #   so there is no need to sum weights
    directed_weighted_graph = nx.read_edgelist(paths['edge_list'], create_using=nx.DiGraph(), comments='%', 
        nodetype=int, data=(('weight', float),))
    nx.write_edgelist(directed_weighted_graph, paths['prepared_edge_list'], delimiter=delimiter)

    # PROCESS ATTRIBUTES
    sorted_nodes = sorted(directed_weighted_graph.nodes)

    if not file_lines(paths['node_attributes']) == len(sorted_nodes):
        raise ValueError('Number of nodes and number of lines in attributes file are not the same')

    with open(paths['node_attributes'], 'r') as source:
        reader = csv.reader(source)
        with open(paths['prepared_node_attributes'], 'w') as result:
            writer = csv.writer(result, delimiter=delimiter)
            writer.writerow(('node_id', 'age'))
            for node_id, attrs in zip(sorted_nodes, reader):
                try:
                    writer.writerow((node_id, attrs[0]))
                except IndexError:
                    writer.writerow((node_id, ''))
    # fill NaN in age
    attrs_df = pd.read_csv(paths['prepared_node_attributes'], delimiter=delimiter)
    attrs_df[['age']] = attrs_df[['age']].apply(lambda x: round(x.fillna(x.mean())))
    attrs_df.to_csv(paths['prepared_node_attributes'], sep=delimiter, index=False)


def prepare_moreno_seventh(dataset_name, edge_list_filename, node_attributes_filename):
    paths = get_paths_dict(dataset_name, edge_list_filename, node_attributes_filename)

    if not os.path.exists(paths['prepared_dataset_dir']):
        os.mkdir(paths['prepared_dataset_dir'])

    # PROCESS EDGES
    directed_weighted_graph = nx.read_edgelist(paths['edge_list'], create_using=nx.DiGraph(), comments='%',
        nodetype=int, data=(('weight', float),))
    nx.write_edgelist(directed_weighted_graph, paths['prepared_edge_list'], delimiter=delimiter)

    # PROCESS ATTRIBUTES
    sorted_nodes = sorted(directed_weighted_graph.nodes)

    if not file_lines(paths['node_attributes']) == len(sorted_nodes):
        raise ValueError('Number of nodes and number of lines in attributes file are not the same')

    with open(paths['node_attributes'], 'r') as source:
        reader = csv.reader(source)
        with open(paths['prepared_node_attributes'], 'w') as result:
            writer = csv.writer(result, delimiter=delimiter)
            writer.writerow(('node_id', 'gender'))
            for node_id, attrs in zip(sorted_nodes, reader):
                writer.writerow((node_id, attrs[0]))


def prepare_petster_hamster(dataset_name, edge_list_filename, node_attributes_filename):
    paths = get_paths_dict(dataset_name, edge_list_filename, node_attributes_filename)

    if not os.path.exists(paths['prepared_dataset_dir']):
        os.mkdir(paths['prepared_dataset_dir'])

    # PROCESS EDGES
    graph = nx.read_edgelist(paths['edge_list'], create_using=nx.Graph(), comments='%',
        nodetype=int)
    # set weights
    nx.set_edge_attributes(graph, name='weight', values=1)
    # to directed
    directed_weighted_graph = graph.to_directed()
    nx.write_edgelist(directed_weighted_graph, paths['prepared_edge_list'], delimiter=delimiter)

    # PROCESS ATTRIBUTES
    sorted_nodes = sorted(directed_weighted_graph)
    # columns
    # ent dat.name dat.joined dat.species dat.coloring dat.gender dat.birthday dat.age dat.hometown dat.favorite_toy dat.favorite_activity dat.favorite_food
    # columns to stay
    # [0] ent - node_id, [3] species, [4] coloring, [5] gender, [7]age?, [8]hometown?
    with open(paths['node_attributes'], 'r', encoding='windows-1250') as source:
        reader = csv.reader(source, delimiter=' ')
        # skip first 3 rows
        for _ in range(3):
            next(reader)
        with open(paths['prepared_node_attributes'], 'w') as result:
            writer = csv.writer(result, delimiter=delimiter)
            writer.writerow(('node_id', 'species', 'coloring', 'gender'))
            for node_id, attrs in zip(sorted_nodes, reader):
                # handle lack of attributes
                while node_id != int(attrs[0]):
                    writer.writerow((node_id, '', '', ''))
                    node_id += 1
                writer.writerow((node_id, attrs[3], attrs[4], attrs[5]))


def prepare_email_eu(dataset_name, edge_list_filename, node_attributes_filename):
    paths = get_paths_dict(dataset_name, edge_list_filename, node_attributes_filename)

    if not os.path.exists(paths['prepared_dataset_dir']):
        os.mkdir(paths['prepared_dataset_dir'])

    # PROCESS EDGES
    directed_graph = nx.read_edgelist(paths['edge_list'], create_using=nx.DiGraph(), nodetype=int)
    # graph is already directed
    # set all weights equal to 1
    nx.set_edge_attributes(directed_graph, name='weight', values=1)
    nx.write_edgelist(directed_graph, paths['prepared_edge_list'], delimiter=delimiter)

    # PROCESS ATTRIBUTES
    sorted_nodes = sorted(directed_graph.nodes)

    if file_lines(paths['node_attributes']) != len(sorted_nodes):
        raise ValueError('Number of nodes and number of lines in attributes file are not the same')
    # replace department numbers with letters -> to prevent casting to numeric 
    #   while loading into dataframe
    attrs_df = pd.read_csv(paths['node_attributes'], delimiter=' ', names=['node_id', 'department'])
    attrs_df = pd.get_dummies(attrs_df, columns=['department'], prefix='department')
    attrs_df.to_csv(paths['prepared_node_attributes'], sep=delimiter, index=False)

# TODO: refactoring

prepare_primary_school('primary_school', 'primaryschool.csv', 'metadata_primaryschool.txt')
prepare_workplace('workplace', 'tij_InVS.dat', 'metadata_InVS13.txt')
prepare_highschool('highschool_2011', 'thiers_2011.csv', 'metadata_2011.txt')
prepare_highschool('highschool_2012', 'thiers_2012.csv', 'metadata_2012.txt')
prepare_hospital('hospital', 'detailed_list_of_contacts_Hospital.dat')
prepare_moreno_blogs('moreno_blogs', 'out.moreno_blogs_blogs', 'ent.moreno_blogs_blogs.blog.orientation')
prepare_moreno_sheep('moreno_sheep', 'out.moreno_sheep_sheep', 'ent.moreno_sheep_sheep.sheep.age')
prepare_moreno_seventh('moreno_seventh', 'out.moreno_seventh_seventh', 'ent.moreno_seventh_seventh.student.gender')
prepare_petster_hamster('petster-hamster', 'out.petster-hamster', 'ent.petster-hamster')
prepare_email_eu('email-Eu', 'email-Eu-core.txt', 'email-Eu-core-department-labels.txt')
print('done')
