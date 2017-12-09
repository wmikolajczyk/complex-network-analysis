import os
import networkx as nx

from utils import PrimarySchoolDatasetHandler
from config import primaryschool

# Prepare dataset file
if not os.path.exists(primaryschool['prepared_dataset']):
    PrimarySchoolDatasetHandler.prepare_dataset(primaryschool['dataset'], primaryschool['prepared_dataset'])
# Read metadata
class_id, gender = PrimarySchoolDatasetHandler.read_metadata(primaryschool['metadata'])
# Create graph with edges loaded from dataset
graph = nx.read_edgelist(primaryschool['prepared_dataset'], nodetype=int)
# Add attributes to nodes
nx.set_node_attributes(graph, class_id, 'class')
nx.set_node_attributes(graph, gender, 'gender')
