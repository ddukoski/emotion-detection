import csv
import numpy as np
import pandas as pd

def read_data(arb_csv_file):
    """
    :param arb_csv_file: A dataset in comma-seperated-values format
    :return: A 2-D numpy array containing the dataset, each picture classified per element of the array
    """
    with open(arb_csv_file) as filename:
        em_iterator = csv.reader(filename)
        return list(em_iterator)


if __name__ == '__main__':
    full_ds = read_data('datasets/fer2013.csv')[1:]

    train = list()
    public_test = list()
    private_test = list()

    for sample in full_ds:
        type_use = sample[-1]
        if type_use == 'Training':
            train.append(sample)
        elif type_use == 'PrivateTest':
            private_test.append(sample)
        else:
            public_test.append(sample)

    training_dataframe = pd.DataFrame(np.array(train))
    priv_test_dataframe = pd.DataFrame(np.array(private_test))
    pub_test_dataframe = pd.DataFrame(np.array(public_test))

    training_dataframe.to_csv('datasets/train.csv', index=False)
    priv_test_dataframe.to_csv('datasets/private_test.csv', index=False)
    pub_test_dataframe.to_csv('datasets/public_test.csv', index=False)
