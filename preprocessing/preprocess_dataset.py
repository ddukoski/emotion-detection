import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from preprocessing_frames import preprocess
from keras_preprocessing.image.image_data_generator import ImageDataGenerator

import os


# TODO explain everything
def preprocess_dataset(dataset_folder, emotion_filenames):
    for emotion in emotion_filenames:
        # concatenate part
        folder_path = os.path.join(dataset_folder, emotion)
        for image in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image)
            train_set_img = cv2.imread(image_path)
            preprocessed_image = preprocess(train_set_img)
            # rewrite original images
            cv2.imwrite(image_path, preprocessed_image)
            expand(preprocessed_image, image_path)


def read_dataset(dataset_path):
    data = pd.read_csv(dataset_path)
    data.columns = ['emotion_num', 'image_px', 'type']
    # Convert each image to a numpy array
    data['image_px'] = data['image_px'].apply(lambda px: np.fromstring(px, sep=' '))
    data['image_px'] = data['image_px'].apply(lambda image: preprocess(image, False))
    return data['image_px']


def expand(image, save_to_datapath):
    datagen = ImageDataGenerator(
        rotation_range=45,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='reflect',
        brightness_range=(0.5, 1.5)
    )


if __name__ == '__main__':
    """""
    The data augmentation process will include addition of random elements like noise, rotation, scaling and cropping on
    already existing images in the test set thus enriching the dataset. 
    We can also use kernel filters for sharpening and blurring. """
    images = read_dataset('fer2013.csv')
    test_img = images[1]
    print(test_img)
    print(test_img.shape)
    plt.imshow(test_img, cmap='gray')
    plt.title("Test image")
    plt.axis('off')
    plt.show()
