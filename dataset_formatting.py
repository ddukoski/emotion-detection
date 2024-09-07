import csv

import cv2
import numpy as np
import pandas as pd
from preprocessing.preprocessing_frames import *


def preprocess_dataset(dataset_path):
    data = pd.read_csv(dataset_path)
    data.columns = ['emotion', 'image', 'type']
    # convert each image to a numpy array
    data['image'] = data['image'].apply(lambda px: np.fromstring(px, sep=' '))
    data['image'] = data['image'].apply(lambda image: preprocess(image, False))

    # convert back to string to save and process correctly in the CSV file
    data['image'] = data['image'].apply(lambda arr: ' '.join(map(str, arr.flatten())))
    # this part is done for simplicity - gets rid of the last column after preprocessing
    data = data.drop(columns=['type'])

    return data


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
        # separation of data types (train, test and validation)
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

    # creates separate datasets as CSV files
    training_dataframe.to_csv('datasets/train.csv', index=False)
    priv_test_dataframe.to_csv('datasets/private_test.csv', index=False)
    pub_test_dataframe.to_csv('datasets/public_test.csv', index=False)

    # TRAIN SET
    preprocessed_train_data = preprocess_dataset('datasets/train.csv')
    preprocessed_train_data.to_csv('datasets/train.csv', index=False)

    # PRIVATE TEST - VALIDATION SET
    preprocessed_private_data = preprocess_dataset('datasets/private_test.csv')
    preprocessed_private_data.to_csv('datasets/private_test.csv', index=False)

    # PUBLIC TEST - TEST SET
    preprocessed_validation_data = preprocess_dataset('datasets/public_test.csv')
    preprocessed_validation_data.to_csv('datasets/public_test.csv', index=False)

    # after this each set contains two columns: emotion and image

