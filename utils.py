import csv

from abc import ABC, abstractmethod


class NetworkDatasetHandler(ABC):
    """
    Abstract class for handling network datasets
    """

    def __init__(self, dataset_file, dest_file, metadata_file, variable):
        self.dataset_file = dataset_file
        self.dest_file = dest_file
        self.metadata_file = metadata_file
        self.variable = variable
        super(NetworkDatasetHandler, self).__init__()

    @abstractmethod
    def read_metadata(self):
        pass

    @abstractmethod
    def prepare_training_dataset(self):
        pass


class PrimarySchoolDatasetHandler:
    """
    http://www.sociopatterns.org/datasets/primary-school-temporal-network-data/
    Prepare primary school dataset
    """

    @staticmethod
    def prepare_graph_dataset(dataset_file, dest_file):
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
    def read_metadata(metadata_file):
        """
        attributes = {
            node_id: {
                'attr1': val1,
                'attr2': val2
            }
        }
        :param metadata_file:
        :return:
        """
        attributes = {}
        with open(metadata_file, 'r') as f:
            for line in f:
                split = line.split()
                node_id = int(split[0])
                attributes[node_id] = {
                    'class': split[1],
                    'gender': split[2]
                }
        return attributes

    @staticmethod
    def prepare_training_dataset(dataset_file, dest_file, gender):
        with open(dataset_file, 'r') as source:
            reader = csv.reader(source, delimiter='\t')
            with open(dest_file, 'w') as result:
                writer = csv.writer(result, delimiter='\t')
                writer.writerow(
                    ('class1', 'gender1', 'class2', 'gender2')
                )
                for row in reader:
                    writer.writerow(
                        (row[3], gender[int(row[1])], row[4],
                         gender[int(row[2])])
                    )


class WorkplaceDatasetHandler:
    """
    http://www.sociopatterns.org/datasets/contacts-in-a-workplace/
    """

    @staticmethod
    def read_metadata(metadata_file):
        department = {}
        with open(metadata_file, 'r') as f:
            for line in f:
                split = line.split()
                department[int(split[0])] = split[1]
        return department

    @staticmethod
    def prepare_training_dataset(dataset_file, dest_file, department):
        with open(dataset_file, 'r') as source:
            reader = csv.reader(source, delimiter=' ')
            with open(dest_file, 'w') as result:
                writer = csv.writer(result, delimiter='\t')
                writer.writerow(
                    ('department1', 'department2')
                )
                for row in reader:
                    writer.writerow(
                        (department[int(row[1])], department[int(row[2])])
                    )
