import cv2
from preprocessing.preprocessing_frames import preprocess

def detect_face(frame, fromvid, min_neighb, scale_fac):
    """
    :param frame: A digital image with a potential human face present
    :return: All detected and preprocessed faces, along with a frame with a drawn rectangle along each human face
    """
    MIN_NEIGHBOURS = 9
    SCALE_FACTOR = 1.1

    face_cascade = cv2.CascadeClassifier('preprocessing/haarcascade_frontalface_default.xml')
    # TODO explore skipping frames for less
    """"
    The code for capturing and preprocessing video frames shall process each n-th frame 
    for performance reasons. Capturing every single frame and preprocessing it to fit well for the dataset is a costly 
    task. We will exploit the possibility of humans detecting  30-60 frames per second. The fps of the video for testing is
    25.0. 
    """

    # remove noise from the image with a median filer with a 5x5 kernel - THIS IS OPTIONAL
    # gray_image = cv2.medianBlur(gray_image, 5)
    """
    Find the region of interest (faces), scale_factor has a regular value
    minNeighbours has a higher value for greater confidence level
    """
    ROI_coordinates = face_cascade.detectMultiScale(frame, scaleFactor=scale_fac, minNeighbors=min_neighb)
    """ after finding the coordinates of each face in the frame, we should extract
    separate regions of interest (single faces)
    using the coordinates specified for width and height for a rectangle shape"""

    preprocessed_faces = []

    for (x, y, w, h) in ROI_coordinates:
        """"
        draw a rectangle around the face
        using the coordinates of the top left and lower right corner of the rectangle
        this part is optional but helpful
        """
        # using the information about width and height we will extract (crop) the face
        # y : y + h is the height
        # x : x + w is the width
        ROI_face = frame[y:y + h, x:x + w]
        # resize the image to fit the size of the images in the training set
        ROI_resized = cv2.resize(ROI_face, (48, 48))

        """
        Its important to note that at this line the frames go under operations regarding enhancement
        / improvement of the ML/AI part. The function used is applied to the original dataset, the only difference
        is that we've enriched the dataset using augmentation
        """
        preprocessed_faces.append(preprocess(ROI_resized, from_video=fromvid))

    return ROI_coordinates, frame, preprocessed_faces


"""
ADDITIONAL COMMENTS
We've done several test on a couple of parameters to see the performance of the Haar Cascade Detector.
The processing is noticeably faster when the scale factor is greater. Initially, it was set at 1.1 (a standard value)
This has proven to be quite slow, as the video looked like it was played on 0.5x speed.
The number of detected faces at this rate was 301, while the total number of correctly read frames was 318.
At ScaleFactor set to 1.3, the same parameters are as follows: 
The number of detected faces at this rate was 301, while the total number of correctly read frames was 301,
while at ScaleFactor set to 1.7, the video was proven to be significantly faster than both, although
the number of detected faces at this rate was 291. AN IMPORTANT NOTE: it would be helpful to 
calculate te scale factor based on the frames resolution.
"""
