import csv
import networkx as nx
import os
import pandas as pd

from utils import PrimarySchoolDatasetHandler
from config import primaryschool, primaryschool_dataset_dir

# Load network 1 dataset

primary_school_handler = PrimarySchoolDatasetHandler()
# Prepare dataset file
if not os.path.exists(primaryschool['prepared_dataset']):
    primary_school_handler.prepare_dataset(primaryschool['dataset'], primaryschool['prepared_dataset'])

# Read metadata
class_id, gender = primary_school_handler.read_metadata(primaryschool['metadata'])

# Create graph with edges loaded from dataset
graph = nx.read_edgelist(primaryschool['prepared_dataset'], nodetype=int)
# Add attributes to nodes
nx.set_node_attributes(graph, class_id, 'class')
nx.set_node_attributes(graph, gender, 'gender')

print(graph.nodes(data=True))

# Prepare csv for dataframe
dest_file = os.path.join(primaryschool_dataset_dir, 'prepared_data.csv')
with open(primaryschool['dataset'], 'r') as source:
    reader = csv.reader(source, delimiter='\t')
    with open(dest_file, 'w') as result:
        writer = csv.writer(result, delimiter='\t')
        writer.writerow(
            ('class1', 'gender1', 'class2', 'gender2')
        )
        for row in reader:
            writer.writerow(
                (row[3], gender[int(row[1])], row[4], gender[int(row[2])])
            )

df = pd.read_csv(dest_file, sep='\t')
print(df)

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
