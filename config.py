import os

# File paths
primaryschool_dataset_dir = 'Datasets/primary_school/'
primaryschool = {
    'dataset': os.path.join(primaryschool_dataset_dir, 'primaryschool.csv'),
    'prepared_graph_dataset': os.path.join(primaryschool_dataset_dir, 'primaryschool_prepared.csv'),
    'metadata': os.path.join(primaryschool_dataset_dir, 'metadata_primaryschool.txt'),
    'prepared_data': os.path.join(primaryschool_dataset_dir, 'prepared_data.csv')
}

workplace_dataset_dir = 'Datasets/workplace'
workplace = {
    'dataset': os.path.join(workplace_dataset_dir, 'tij_InVS.dat'),
    'metadata': os.path.join(workplace_dataset_dir, 'metadata_InVS13.txt'),
    'prepared_data': os.path.join(workplace_dataset_dir, 'prepared_data.csv')
}
