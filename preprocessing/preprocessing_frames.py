import cv2
import numpy as np

# haar cascade is a library used for object detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# this script will be used to transform the training, validation and test set
# the results will be saved and used to train the CNN
def preprocess(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    """ adaptive histogram points out subtle differences better than histogram equalization
    this is because it is focuses on distinct locations of the frame, rather than the frame globally
    as a result, specific facial landmarks/expressions are more distinct
    first we should create a CLAHE object
    (Contrast Limited Adaptive Histogram Equalization)
    clapLimit controls the limit of the contrast and with a value of 2.0 it won't over accentuate,
    We also apply image sharpening for accentuating edges better
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    final_frame = clahe.apply(frame)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    final_frame = cv2.filter2D(final_frame, -1, kernel)

    # TODO: normalization!
    final_frame = final_frame / 255.0

    return final_frame


def draw_rectangle_around(frame):
    """
    Draw a rectangle around the detected face using a built-in function in cv2
    :param frame: successfully loaded frame from one instance of the video
    :return: detected faces in BGR format with 48x48 resolution
    """
    faces = []
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # remove noise from the image with a median filer with a 5x5 kernel - THIS IS OPTIONAL
    # gray_image = cv2.medianBlur(gray_image, 5)
    """
    Find the region of interest (faces), scale_factor has a regular value
    minNeighbours has a higher value for greater confidence level
    """
    ROI_coordinates = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=9)
    """ after finding the coordinates of each face in the frame, we should extract
    separate regions of interest (single faces)
    using the coordinates specified for width and height for a rectangle shape"""
    for (x, y, w, h) in ROI_coordinates:
        """"
        Draw a rectangle around the face using the coordinates of the top left and lower right corner of the rectangle.
        this part is optional but helpful
        """
        cv2.rectangle(frame, (x, y), (x + w, y + h), (225, 105, 65), 2)
        # Using the information about width and height we will extract (crop) the face from the frame
        # This will create a separate image
        # y : y + h is the height
        # x : x + w is the width
        ROI_face = frame[y:y + h, x:x + w]
        # Resize the image to fit the size of the images in the training set
        ROI_resized = cv2.resize(ROI_face, (48, 48))
        faces.append(ROI_resized)
        """
                   Its important to note that at this line the frames go under operations regarding enhancement
                   / improvement of the ML/AI part. The function used is applied to the original dataset, the only difference
                   is that we've enriched the dataset using augmentation
                   """

        return faces
