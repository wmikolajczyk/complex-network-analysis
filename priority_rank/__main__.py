import networkx as nx
import os

from utils import PrimarySchoolDatasetHandler
from config import primaryschool_prepared_dataset, primaryschool_dataset, primaryschool_metadata

# Load network 1 dataset

primary_school_handler = PrimarySchoolDatasetHandler()
# Prepare dataset file
if not os.path.exists(primaryschool_prepared_dataset):
    primary_school_handler.prepare_dataset(primaryschool_dataset, primaryschool_prepared_dataset)

# Read metadata
class_id, gender = primary_school_handler.read_metadata(primaryschool_metadata)

# Create graph with edges loaded from dataset
graph = nx.read_edgelist(primaryschool_prepared_dataset, nodetype=int)
# Add attributes to nodes
nx.set_node_attributes(graph, class_id, 'class')
nx.set_node_attributes(graph, gender, 'gender')

print(graph.nodes(data=True))


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
