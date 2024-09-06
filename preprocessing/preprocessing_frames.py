import cv2
import numpy as np


# this script will be used to transform the training, validation and test set
# the results will be saved and used to train the CNN
def preprocess(frame, from_video):
    if from_video:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        frame = frame.reshape(48, 48, 1).astype(np.uint8)
    """ adaptive histogram points out subtle differences better than histogram equalization
    this is because it is focuses on distinct locations of the frame, rather than the frame globally
    as a result, specific facial landmarks/expressions are more pronounced
    first we should create a CLAHE object
    (Contrast Limited Adaptive Histogram Equalization)
    clapLimit controls the limit of the contrast and with a value of 2.0 it won't over accentuate,
    We also apply image sharpening for accentuating edges better
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    final_frame = clahe.apply(frame)
    # kernel for sharpening the image is applied (optional), also points out edges and angles
    # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # final_frame = cv2.filter2D(final_frame, -1, kernel)

    # TODO: normalization, necessary step to avoid saturation in a neural network
    final_frame = final_frame / 255.0

    return final_frame
