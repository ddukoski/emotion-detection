import Augmentor
import numpy as np
import pandas as pd
from PIL import Image
import os


def save_image_from_row(row, index):
    emotion = row['emotion']
    pixels = row['pixels']

    pixel_array = np.array([int(p) for p in pixels.split()]).reshape(48, 48)
    pixel_array = pixel_array.astype(np.uint8)
    img = Image.fromarray(pixel_array)
    # we must include the index of the image to create a unique name in the folder
    img_filename = os.path.join('aug_images', f"{emotion}_emotion_{index}.jpg")
    img.save(img_filename)

def find_emotion(filename):
    number_of_emotion = 0

def create_set_from_jpg(directory_path):
    for image in os.listdir(directory_path):
        emotion = find_emotion(image)
        



if __name__ == '__main__':
    read_pixels = pd.read_csv('train.csv')
    if not os.path.exists('aug_images'):
        os.makedirs('aug_images')

    # for index, row in read_pixels.iterrows():
    #     save_image_from_row(row, index)

    p = Augmentor.Pipeline(source_directory='aug_images', output_directory='aug_images')
    p.rotate(probability=0.2, max_left_rotation=25, max_right_rotation=25)
    p.skew(probability=0.2, magnitude=0.2)
    p.zoom_random(probability=0.3, percentage_area=0.8)
    p.gaussian_distortion(
        probability=0.2,
        grid_width=4,
        grid_height=4,
        magnitude=8,
        corner="bell",
        method="in"
    )
    p.random_erasing(probability=0.1, rectangle_area=0.2)
    p.sample(1000)
    print('Images are augmented.')

    augmented_images_data= 'C:\\Users\\janaa\\PycharmProjects\\emotion-det\\datasets\\aug_images\\aug_images'
    train_set_images_path = 'C:\\Users\janaa\\PycharmProjects\\emotion-det\\datasets\\aug_images'
    train_dataset_csv_path = 'C:\\Users\\janaa\\PycharmProjects\\emotion-det\\datasets\\train.csv'






