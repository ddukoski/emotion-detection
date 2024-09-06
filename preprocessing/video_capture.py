import cv2
import numpy as np
from preprocessing_frames import preprocess

# TODO load the model for prediction

# TODO find sample videos and write PATHS
video_files = ['C:\\Users\\janaa\\Videos\\reading.mp4']
video = video_files[0]
capture = cv2.VideoCapture(video)
emotions = ['happy', 'sad', 'surprised', 'angry', 'scared']


if not capture.isOpened():
    print("Error opening video stream or file")

# haar cascade is a library used for object detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# TODO explore skipping frames for less
""""
The code for capturing and preprocessing video frames shall process each n-th frame 
for performance reasons. Capturing every single frame and preprocessing it to fit well for the dataset is a costly 
task. We will exploit the possibility of humans detecting  30-60 frames per second. The fps of the video for testing is
25.0. 
"""
while capture.isOpened():
    flag, frame = capture.read()
    frame = cv2.resize(frame, (640, 400))

    if not flag:
        print("Error reading the frame")
        break
    else:
        # transform to grayscale for simpler calculations
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
            draw a rectangle around the face
            using the coordinates of the top left and lower right corner of the rectangle
            this part is optional but helpful
            """
            cv2.rectangle(frame, (x, y), (x + w, y + h), (225, 105, 65), 4)
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
            preprocessed_image = preprocess(ROI_resized)
            cv2.imshow('Preprocessed Frame', preprocessed_image)

        cv2.imshow('Video Frame', frame)
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
capture.release()
cv2.destroyAllWindows()

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
