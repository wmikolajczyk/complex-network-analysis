import csv


def prepare_primaryschool_dataset(source_file, dest_file):
    """
    http://www.sociopatterns.org/datasets/primary-school-temporal-network-data/
    Leave only node edges data
    :return:
    """
    with open(source_file, 'r') as source:
        reader = csv.reader(source, delimiter='\t')
        with open(dest_file, 'w') as result:
            writer = csv.writer(result, delimiter='\t')
            for row in reader:
                writer.writerow((row[1], row[2]))

