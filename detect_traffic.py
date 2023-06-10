import cv2  # Computer vision library
import numpy as np  # Scientific computing library
from Utils import object_detection   # Custom object detection program
from tensorflow import keras  # Library for neural networks
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

# Detect traffic light color in a batch of image files
files = object_detection.get_files('test_images/*.jpg')

# Load the SSD neural network that is trained on the COCO data set
model_ssd = object_detection.load_ssd_coco()

# Load the trained neural network
model_traffic_lights_nn = keras.models.load_model("traffic.h5")

# Go through all image files, and detect the traffic light color.
for file in files:
    (img, out, file_name) = object_detection.perform_object_detection(
        model_ssd, file, save_annotated=True, model_traffic_lights=model_traffic_lights_nn)
