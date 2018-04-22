import networkx as nx
import os
import re
import requests
import tarfile
import wget

from bs4 import BeautifulSoup

from generate_graph import attach_attributes

dataset_names = [
    'moreno_crime',
    'unicodelang'
]


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


available_datasets = get_available_datasets()


def get_network_from_konect(network_name):
    downloaded_filepath = download_dataset(network_name)
    extracted_filepath = extract_dataset(downloaded_filepath, network_name)
    graph = load_graph(extracted_filepath)
    return graph


def download_dataset(network_name):
    try:
        dataset = available_datasets[network_name]
    except KeyError:
        print('Dataset does not exist')
        return

    download_dir = 'downloads'
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    filepath = os.path.join(download_dir, dataset['filename'])
    wget.download(dataset['url'], out=filepath)
    return filepath


def extract_dataset(filepath, network_name):
    datasets_dir = 'networks'
    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir)

    with tarfile.open(filepath) as archive:
        archive.extractall(datasets_dir)

    return os.path.join(datasets_dir, network_name)


def file_gen(f_name):
        with open(f_name) as f:
            for line in f:
                yield line.rstrip()


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
        import pdb; pdb.set_trace()
        raise ValueError('There should be exactly one out. file')
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
        ent_gens.append(file_gen(ent_filepath))

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


for dataset_name in dataset_names:
    graph = get_network_from_konect(dataset_name)
    import pdb; pdb.set_trace()
