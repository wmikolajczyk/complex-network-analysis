import csv


class PrimarySchoolDatasetHandler:
    """
    http://www.sociopatterns.org/datasets/primary-school-temporal-network-data/
    Prepare primary school dataset
    """

    @staticmethod
    def prepare_dataset(dataset_file, dest_file):
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

        :param metadata_file:
        :return:
        """
        class_id = {}
        gender = {}
        with open(metadata_file, 'r') as f:
            for line in f:
                split = line.split()
                class_id[int(split[0])] = split[1]
                gender[int(split[0])] = split[2]
        return class_id, gender

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
