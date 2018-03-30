import csv
import pandas as pd

from abc import ABC, abstractmethod


class NetworkDatasetHandler(ABC):
    """
    Abstract class for handling network datasets
    """

    @staticmethod
    def read_metadata(metadata_file, attribute_names):
        attributes = {}
        with open(metadata_file, 'r') as f:
            for line in f:
                split = line.split()
                node_id = int(split[0])

                attributes[node_id] = {
                    attribute: split[i]
                    for i, attribute in enumerate(attribute_names, 1)
                }
        return attributes

    @abstractmethod
    def prepare_training_dataset(self):
        pass


class PrimarySchoolDatasetHandler(NetworkDatasetHandler):
    """
    http://www.sociopatterns.org/datasets/primary-school-temporal-network-data/
    Prepare primary school dataset
    """

    @staticmethod
    def export_edges(dataset_file, dest_file):
        """
        Leave only node edges data
        :return:
        """
        with open(dataset_file, 'r') as source:
            reader = csv.reader(source, delimiter='\t')
            with open(dest_file, 'w') as result:
                writer = csv.writer(result, delimiter='\t')
                for row in reader:
                    writer.writerow((row[1], row[2]))


    @staticmethod
    def export_node_connections(nodes_list, adj_matrix_tril, dest_file):
        """
        For debugging

        Create csv with node ids and num of edges between nodes
        it gives an output with 29161 rows which is ok because
        242^2 = 29282
        29282 - 29161 = 121
        121 is 242 (number of nodes) / 2
        because we take lower triangle of matrix
        :return:
        """
        with open(dest_file, 'w') as result:
            writer = csv.writer(result, delimiter='\t')
            for i in range(1, len(nodes_list)):
                node1_id = nodes_list[i]
                for j in range(i):
                    node2_id = nodes_list[j]
                    num_of_edges = adj_matrix_tril[i][j]
                    writer.writerow((node1_id, node2_id, num_of_edges))

    @staticmethod
    def export_node_connections_attributes(nodes_list, node_attributes, adj_matrix_tril, dest_file):
        """
        Create csv with node attributes and num of edges between nodes
        :return:
        """
        with open(dest_file, 'w') as result:
            writer = csv.writer(result, delimiter='\t')
            writer.writerow(
                ('class1', 'gender1', 'class2', 'gender2', 'num_of_connections')
            )
            for i in range(1, len(nodes_list)):
                node1_id = nodes_list[i]
                node1_attrs = list(node_attributes[node1_id].values())
                for j in range(i):
                    node2_id = nodes_list[j]
                    node2_attrs = list(node_attributes[node2_id].values())
                    num_of_edges = adj_matrix_tril[i][j]
                    writer.writerow((
                        node1_attrs[0], node1_attrs[1],
                        node2_attrs[0], node2_attrs[1],
                        num_of_edges
                    ))

    @staticmethod
    def clean_data(primaryschool_df):
        #   - Transform data to training dataset
        #       remove 'Unknown' values in gender columns
        female1 = primaryschool_df['gender1'].value_counts()['F']
        female1_prob = female1 / primaryschool_df.shape[0]
        gender_to_replace = ['M', 'F'][female1_prob >= 0.5]
        primaryschool_df['gender1'] = primaryschool_df['gender1'].replace(
            'Unknown', gender_to_replace)

        female2 = primaryschool_df['gender2'].value_counts()['F']
        female2_prob = female2 / primaryschool_df.shape[0]
        gender_to_replace = ['M', 'F'][female2_prob >= 0.5]
        primaryschool_df['gender2'] = primaryschool_df['gender2'].replace(
            'Unknown', gender_to_replace)

        #   - Mark gender columns as categorical and apply encoding
        primaryschool_df['gender1'] = primaryschool_df['gender1'].astype(
            'category')
        primaryschool_df['gender2'] = primaryschool_df['gender2'].astype(
            'category')

        cat_columns = primaryschool_df.select_dtypes(['category']).columns
        primaryschool_df[cat_columns] = primaryschool_df[cat_columns].apply(
            lambda x: x.cat.codes)

        #   - Create dummies from class categorical columns which have more than 2 different values
        primaryschool_df = pd.get_dummies(primaryschool_df,
                                          columns=['class1', 'class2'])

        #   - Change dataframe column ordering (move num_of_connections to the last index)
        cols = primaryschool_df.columns.tolist()
        cols[2], cols[len(cols) - 1] = cols[len(cols) - 1], cols[2]

        return primaryschool_df[cols]


class WorkplaceDatasetHandler(NetworkDatasetHandler):
    """
    http://www.sociopatterns.org/datasets/contacts-in-a-workplace/
    """

    @staticmethod
    def prepare_training_dataset(dataset_file, dest_file, attributes):
        with open(dataset_file, 'r') as source:
            reader = csv.reader(source, delimiter=' ')
            with open(dest_file, 'w') as result:
                writer = csv.writer(result, delimiter='\t')
                writer.writerow(
                    ('department1', 'department2')
                )
                for row in reader:
                    writer.writerow(
                        (attributes[int(row[1])]['department'], attributes[int(row[2])]['department'])
                    )
