import os
import requests
import tarfile
import wget
import pickle

from bs4 import BeautifulSoup
from collections import defaultdict
from shutil import copyfile

base_url = 'http://konect.uni-koblenz.de/downloads/'
response = requests.get(base_url)
soup = BeautifulSoup(response.text, 'html.parser')

table = soup.find(id='sort1')
tbody = table.find('tbody')

dataset_partial_links = []
for row in tbody.findAll('tr'):
    for link in row.findAll('a'):
        url = link.get('href')
        if 'tsv' in url:
            dataset_partial_links.append(url)

dataset_links = [base_url + partial_link for partial_link in dataset_partial_links]

dl_dirname = 'downloaded'
datasets_with_attrs = defaultdict(list)
selected_dirname = 'selected'


def download_all():
    max_filesize = 100000000
    if not os.path.exists(dl_dirname):
            os.makedirs(dl_dirname)

    for link in dataset_links:
        filename = link.split('/')[-1]
        filepath = os.path.join(dl_dirname, filename)
        if os.path.exists(filepath):
            print('skipping {} - already exists'.format(filename))
            continue

        if int(requests.head(link).headers['content-length']) > max_filesize:
            print('skipping dataset - larger than {}b'.format(max_filesize))
            continue

        print('downloading {}'.format(filename))
        wget.download(link, out=filepath)


def find_datasets_with_attrs():
    for filename in os.listdir(dl_dirname):
        print('processing: {}'.format(filename))
        filepath = os.path.join(dl_dirname, filename)
        with tarfile.open(filepath) as archive:
            count = len([member for member in archive if member.isreg()])
        if count > 3:
            datasets_with_attrs[count].append(filepath)
    print(datasets_with_attrs)
    with open('datasets_with_attrs.pkl', 'wb') as file:
        pickle.dump(datasets_with_attrs, file)


def select_datasets():
    if not os.path.exists(selected_dirname):
            os.makedirs(selected_dirname)

    with open('datasets_with_attrs.pkl', 'rb') as file:
        datasets_with_attrs = pickle.load(file)

    for filepath in datasets_with_attrs[5]:
        # copyfile(filepath, os.path.join(selected_dirname, filepath.split('/')[-1]))
        with tarfile.open(filepath) as archive:
            archive.extractall('best_graphs')

select_datasets()
