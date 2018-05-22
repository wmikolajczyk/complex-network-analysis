import os
import requests
import tarfile
import wget
import re
import networkx as nx

from bs4 import BeautifulSoup


def get_available_datasets():
    """
    Get available datasets list
    parse konect table to get datasets info
    """
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
                    'filename': re.search('tsv/([^/]+)', partial_url).group(1),
                    'url': base_url + partial_url
                }
    return available_datasets


def download_dataset(network_name, available_datasets):
    try:
        dataset = available_datasets[network_name]
    except KeyError:
        print('Dataset doest not exist')
        return

    max_filesize = 1000000000  # 1GB
    if int(requests.head(dataset['url']).headers['content-length']) > max_filesize:
        print('Skipping {} - larger than {}b'.format(network_name, max_filesize))
        return

    download_dir = 'downloads'
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    filepath = os.path.join(download_dir, dataset['filename'])
    if not os.path.exists(filepath):
        wget.download(dataset['url'], out=filepath)
    else:
        print('Skipping download {} - already exists'.format(network_name))
    return filepath


def extract_dataset(network_name, archive_filepath):
    dataset_dir = 'networks'
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    network_filepath = os.path.join(dataset_dir, network_name)
    if not os.path.exists(network_filepath):
        with tarfile.open(archive_filepath) as archive:
            archive.extractall(dataset_dir)
    else:
        print('Skipping extracting {} - already exists'.format(network_name))
    return network_filepath


def count_attributes(network_filepath):
    files = os.listdir(network_filepath)
    attribute_files_num = 0
    for filename in files:
        if 'ent.' in filename:
            attribute_files_num += 1
    return attribute_files_num


def load_graph(network_filepath):
    """
    Files in network dir
        out. - adjacency matrix
        ent. - node attributes ordered by node_id
        README, meta - skip
    """
    files = os.listdir(network_filepath)
    out_files = []
    for filename in files:
        if 'out.' in filename:
            out_files.append(filename)

    if len(out_files) != 1:
        print('There should be exactly one out. file')
        return

    adj_file = out_files[0]
    adj_filepath = os.path.join(network_filepath, adj_file)
    graph = nx.read_adjlist(adj_filepath, comments='%', nodetype=int)
    return graph
