import os

# File paths
primaryschool_dataset_dir = 'Datasets/primary_school/'
primaryschool = {
    'dataset': os.path.join(primaryschool_dataset_dir, 'primaryschool.csv'),
    'metadata': os.path.join(primaryschool_dataset_dir, 'metadata_primaryschool.txt'),
    'edges': os.path.join(primaryschool_dataset_dir, 'prepared/edges.csv'),
    'prepared_dataset': os.path.join(primaryschool_dataset_dir, 'prepared/node_connections_attributes.csv')
}

workplace_dataset_dir = 'Datasets/workplace'
workplace = {
    'dataset': os.path.join(workplace_dataset_dir, 'tij_InVS.dat'),
    'metadata': os.path.join(workplace_dataset_dir, 'metadata_InVS13.txt'),
    'prepared_dataset': os.path.join(workplace_dataset_dir, 'prepared/data.csv')
}
