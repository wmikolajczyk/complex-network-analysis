import os

# File paths
primaryschool_dataset_dir = 'Datasets/primary_school/'
primaryschool = {
    'dataset': os.path.join(primaryschool_dataset_dir, 'primaryschool.csv'),
    'prepared_dataset': os.path.join(primaryschool_dataset_dir, 'primaryschool_prepared.csv'),
    'metadata': os.path.join(primaryschool_dataset_dir, 'metadata_primaryschool.txt')
}
