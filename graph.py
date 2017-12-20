import os
import networkx as nx

from utils import PrimarySchoolDatasetHandler
from config import primaryschool

# Prepare dataset file
if not os.path.exists(primaryschool['prepared_graph_dataset']):
    PrimarySchoolDatasetHandler.prepare_graph_dataset(primaryschool['dataset'], primaryschool['prepared_graph_dataset'])
# Read metadata
attributes = PrimarySchoolDatasetHandler.read_metadata(primaryschool['metadata'])
# Create graph with edges loaded from dataset
graph = nx.read_edgelist(primaryschool['prepared_graph_dataset'], nodetype=int)
# TODO: update and fix
# Add attributes to nodes
# nx.set_node_attributes(graph, class_id, 'class')
# nx.set_node_attributes(graph, gender, 'gender')
