import csv


class PrimarySchoolDatasetHandler:
    """
    Prepare primary school dataset
    """

    @staticmethod
    def prepare_dataset(dataset_file, dest_file):
        """
        http://www.sociopatterns.org/datasets/primary-school-temporal-network-data/
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
