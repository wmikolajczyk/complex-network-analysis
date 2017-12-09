import os
import pandas as pd

from utils import PrimarySchoolDatasetHandler, WorkplaceDatasetHandler
from config import primaryschool, primaryschool_dataset_dir

# Load Primary School dataset

# Read metadata
class_id, gender = PrimarySchoolDatasetHandler.read_metadata(primaryschool['metadata'])

# Prepare csv for dataframe
dest_file = os.path.join(primaryschool_dataset_dir, 'prepared_data.csv')
PrimarySchoolDatasetHandler.prepare_training_dataset(
    primaryschool['dataset'], dest_file, gender)
primaryschool_df = pd.read_csv(dest_file, sep='\t')

# Load Workplace dataset
# Read metadata
department = WorkplaceDatasetHandler.read_metadata('Datasets/workplace/metadata_InVS13.txt')

# Prepare csv for dataframe
dest_file2 = 'Datasets/workplace/prepared_data.csv'
WorkplaceDatasetHandler.prepare_training_dataset(
    'Datasets/workplace/tij_InVS.dat', dest_file2, department)
workplace_df = pd.read_csv(dest_file2, sep='\t')

# Transform network to training dataset

# Priority Rank
"""
for each vertex n
    compute ranking R (m-lenght)
    for number of edges for vertex
        sample vertex t from the ranking
        add edge vertex - vertex t
"""
