import csv

import cv2
import numpy as np
import pandas as pd
from preprocessing.preprocessing_frames import *


def preprocess_dataset(dataset_path):
    data = pd.read_csv(dataset_path)
    data.columns = ['emotion', 'image']
    data['image'] = data['image'].apply(lambda px: np.fromstring(px, sep=' '))
    data['image'] = data['image'].apply(lambda image: preprocess(image, False))
    data['image'] = data['image'].apply(lambda arr: ' '.join(map(str, arr.flatten())))
    return data


def read_data(arb_csv_file):
    """
    :param arb_csv_file: A datasets in comma-seperated-values format
    :return: A 2-D numpy array containing the datasets, each picture classified per element of the array
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
        emotion = sample[0]
        pixels = sample[1]
        type_use = sample[-1]

        if type_use == 'Training':
            train.append([emotion, pixels])
        elif type_use == 'PrivateTest':
            private_test.append([emotion, pixels])
        else:
            public_test.append([emotion, pixels])

    training_dataframe = pd.DataFrame(train, columns=['emotion', 'pixels'])
    priv_test_dataframe = pd.DataFrame(private_test, columns=['emotion', 'pixels'])
    pub_test_dataframe = pd.DataFrame(public_test, columns=['emotion', 'pixels'])

    training_dataframe.to_csv('datasets/train.csv', index=False)
    priv_test_dataframe.to_csv('datasets/private_test.csv', index=False)
    pub_test_dataframe.to_csv('datasets/public_test.csv', index=False)

    # preprocessed_train_data = preprocess_dataset('datasets/train.csv')
    # preprocessed_train_data.to_csv('datasets/train.csv', index=False)
    #
    # preprocessed_private_data = preprocess_dataset('datasets/private_test.csv')
    # preprocessed_private_data.to_csv('datasets/private_test.csv', index=False)
    #
    # preprocessed_validation_data = preprocess_dataset('datasets/public_test.csv')
    # preprocessed_validation_data.to_csv('datasets/public_test.csv', index=False)

