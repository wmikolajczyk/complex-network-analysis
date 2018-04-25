import networkx as nx
import os
import re
import requests
import tarfile
import wget
import pickle

from bs4 import BeautifulSoup
from collections import defaultdict

from generate_graph import attach_attributes


# petster-hamster? - required preprocessing data
CHOOSEN_DATASETS = {
    # number_of_attrs: [network_name1, network_name2, ...]
    1: [
        'adjnoun_adjacency', 'advogato', 'cfinder-google', 'discogs_lgenre',
        'moreno_blogs', 'moreno_seventh', 'moreno_sheep', 'subelj_cora',
        'subelj_euroroad', 'subelj_jdk', 'subelj_jung-j'
    ],
    2: [
        'github', 'moreno_crime'
    ]
}


def get_available_datasets():
    base_url = 'http://konect.uni-koblenz.de/downloads/'
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    table = soup.find(id='sort1')
    tbody = table.find('tbody')

    available_datasets = {}
    for row in tbody.findAll('tr'):
        for link in row.findAll('a'):
            partial_url = link.get('href')
            if 'tsv' in partial_url:
                dataset_name = re.search('([^/]+).tar.bz2', partial_url).group(1)
                available_datasets[dataset_name] = {
                    'filename': re.search('tsv/([^/]+)', partial_url)[1],
                    'url': base_url + partial_url
                }
    return available_datasets


def get_network_from_konect(network_name):
    available_datasets = get_available_datasets()
    downloaded_filepath = download_dataset(network_name, available_datasets)
    if not downloaded_filepath:
        return
    extracted_filepath = extract_dataset(downloaded_filepath, network_name)
    graph = load_graph(extracted_filepath)
    return graph


def download_dataset(network_name, available_datasets):
    try:
        dataset = available_datasets[network_name]
    except KeyError:
        print('Dataset does not exist')
        return

    max_filesize = 10000000  # 10MB
    if int(requests.head(dataset['url']).headers['content-length']) > max_filesize:
        print('skipping {} - larger than {}b'.format(network_name, max_filesize))
        return

    download_dir = 'downloads'
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    filepath = os.path.join(download_dir, dataset['filename'])
    if not os.path.exists(filepath):
        wget.download(dataset['url'], out=filepath)
    else:
        print('skipping {} - already exists'.format(network_name))
    return filepath


def extract_dataset(filepath, network_name):
    datasets_dir = 'networks'
    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir)

    dest_filepath = os.path.join(datasets_dir, network_name)
    if not os.path.exists(dest_filepath):
        with tarfile.open(filepath) as archive:
            archive.extractall(datasets_dir)
    else:
        print('skipping {} - already exists'.format(network_name))
    return dest_filepath


def file_gen(f_name, encoding='utf-8'):
        with open(f_name, encoding=encoding) as f:
            for line in f:
                yield line.rstrip()


def file_len(f_name, encoding='utf-8'):
    with open(f_name, encoding=encoding) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def load_graph(extracted_filepath):
    """
    out - adjacency matrix
    ent - node attributes ordered by node id
    README, meta - skip
    rel - relation? skip
    """
    files = os.listdir(extracted_filepath)
    out_files = []
    ent_files = []
    for filename in files:
        if 'out.' in filename:
            out_files.append(filename)
        elif 'ent.' in filename:
            ent_files.append(filename)

    if len(out_files) != 1:
        print('There should be exactly one out. file')
        print(out_files)
        return
    adj_file = out_files[0]

    print('Loading graph...')
    graph = nx.read_adjlist(os.path.join(extracted_filepath, adj_file), comments='%')
    graph = nx.convert_node_labels_to_integers(graph)

    print('Attaching attributes (graph measurements)...')
    attach_attributes(graph)

    print('Attaching real attributes...')

    ent_gens = []
    for ent in ent_files:
        ent_filepath = os.path.join(extracted_filepath, ent)
        # append when there are attribute for each node
        try:
            if file_len(ent_filepath) == graph.number_of_nodes():
                ent_gens.append(file_gen(ent_filepath))
            else:
                print('Attribute file error {}'.format(ent))
        except UnicodeDecodeError:
            print('Trying windows-1250 encoding..')
            if file_len(ent_filepath, encoding='windows-1250') == graph.number_of_nodes():
                ent_gens.append(file_gen(ent_filepath, encoding='windows-1250'))
            else:
                print('Attribute file error {}'.format(ent))

    # each line in ent. file contains node attribute (ordered)
    # so it's very important to ensure that graph.nodes are in ascending ordering
    if not list(graph.nodes) == list(range(graph.number_of_nodes())):
        raise ValueError('Graph nodes aren\'t in ascending order')

    for node_id in graph.nodes:
        node_attributes = {
            filename.split('.')[-1]: next(gen)
            for (filename, gen) in zip(ent_files, ent_gens)
        }
        graph.node[node_id].update(node_attributes)

    return graph


def get_num_of_attrs_for_datasets(available_datasets):
    checked_dataset_with_attrs = defaultdict(list)

    for dataset_name in available_datasets:
        print('Processing {}'.format(dataset_name))
        graph = get_network_from_konect(dataset_name)
        if not graph:
            print('no graph.')
            continue
        # check first node
        num_of_attrs = len(graph.nodes[0])
        if num_of_attrs:
            print(dataset_name)
            checked_dataset_with_attrs[num_of_attrs].append(dataset_name)

        # print(checked_dataset_with_attrs)
        with open('best_datasets.pkl', 'wb') as file:
            pickle.dump(checked_dataset_with_attrs, file)