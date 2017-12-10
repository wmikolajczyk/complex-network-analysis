import pandas as pd

from utils import PrimarySchoolDatasetHandler, WorkplaceDatasetHandler
from config import primaryschool, workplace

# Load Primary School dataset
# Read metadata
class_id, gender = PrimarySchoolDatasetHandler.read_metadata(primaryschool['metadata'])

# Prepare csv for dataframe
PrimarySchoolDatasetHandler.prepare_training_dataset(
    primaryschool['dataset'], primaryschool['prepared_data'], gender)
primaryschool_df = pd.read_csv(primaryschool['prepared_data'], sep='\t')

# Load Workplace dataset
# Read metadata
department = WorkplaceDatasetHandler.read_metadata(workplace['metadata'])

# Prepare csv for dataframe4
WorkplaceDatasetHandler.prepare_training_dataset(
    'Datasets/workplace/tij_InVS.dat', workplace['prepared_data'], department)
workplace_df = pd.read_csv(workplace['prepared_data'], sep='\t')


# Transform network to training dataset
# Primary school
primaryschool_df['class1'] = primaryschool_df['class1'].astype('category')
primaryschool_df['gender1'] = primaryschool_df['gender1'].astype('category')
primaryschool_df['class2'] = primaryschool_df['class2'].astype('category')
primaryschool_df['gender2'] = primaryschool_df['gender2'].astype('category')

cat_columns = primaryschool_df.select_dtypes(['category']).columns
primaryschool_df[cat_columns] = primaryschool_df[cat_columns].apply(lambda x: x.cat.codes)

X = primaryschool_df.iloc[:, 0:2]
Y = primaryschool_df.iloc[:, 2:]


# Priority Rank
"""
for each vertex n
    compute ranking R (m-lenght)
    for number of edges for vertex
        sample vertex t from the ranking
        add edge vertex - vertex t
"""
