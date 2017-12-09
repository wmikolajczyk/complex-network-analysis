import networkx as nx
import os

from utils import prepare_primaryschool_dataset
# Load network 1 dataset
primaryschool_dataset_dir = 'Datasets/primary_school/'
primaryschool_dataset = os.path.join(primaryschool_dataset_dir, 'primaryschool.csv')
primaryschool_prepared_dataset = os.path.join(primaryschool_dataset_dir, 'primaryschool_prepared.csv')

if not os.path.exists(primaryschool_prepared_dataset):
    prepare_primaryschool_dataset(primaryschool_dataset, primaryschool_prepared_dataset)

graph = nx.read_edgelist(primaryschool_prepared_dataset)
# Load network 2 dataset

# Transform network to training dataset

# Priority Rank
"""
for each vertex n
    compute ranking R (m-lenght)
    for number of edges for vertex
        sample vertex t from the ranking
        add edge vertex - vertex t
"""
