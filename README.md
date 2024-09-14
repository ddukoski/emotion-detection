# Welcome.

![eyelogo](https://github.com/user-attachments/assets/d7749b2e-0d8c-48d5-80f1-e21805bef3b1)

##### Empathy is a small computer vision and deep learning project for the Digital Image Processing course at FCSE. Empathy detects faces (with Haar Cascades) in video and images and describes their emotion using a trained/to-be-trained CNN.

The app comes with a preloaded and trained model.

Required packages:
NumPy (numpy), PyTorch (torch), Keras preprocessing library (keras-preprocessing), OpenCV (opencv-python) for Python, tkinter, Pillow (pillow) and Augmentor.

If you would like to train the model from scratch, please download the FER2013 dataset from https://www.kaggle.com/datasets/deadskull7/fer2013 and put it in a 'datasets' directory, then do the following:
  1. Run the preprocessing/dataset_formatting.py script (if you are using a different dataset than FER2013, make sure it matches FER2013's column and emotion number format)
  2. Run the preprocessing/augumentation.py stript
  3. Run the preprocessing/training.py script 

Authors:
Jana Angjelkoska, David Dukoski
