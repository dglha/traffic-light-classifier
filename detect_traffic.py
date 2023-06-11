import cv2  # Computer vision library
import numpy as np  # Scientific computing library
from Utils import object_detection, imagetoarraypreprocessor, simplepreprocessor, simpledatasetloader   # Custom object detection program
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
# for file in files:
#     (img, out, file_name) = object_detection.perform_object_detection(
#         model_ssd, file, save_annotated=True, model_traffic_lights=model_traffic_lights_nn)

def detectTraffic(file):
    (img, out, file_name) = object_detection.perform_object_detection(model_ssd, file, save_annotated=True, model_traffic_lights=model_traffic_lights_nn)
    print(file_name)
    return file_name

classLabels = ["green", "off", "red", "yellow"]
sp = simplepreprocessor.SimplePreprocessor(224, 224) # Thiết lập kích thước ảnh 32 x 32
iap = imagetoarraypreprocessor.ImageToArrayPreprocessor(dataFormat="channels_last") # Gọi hàm để chuyển ảnh sang mảng
sdl = simpledatasetloader.SimpleDatasetLoader(preprocessors=[sp, iap])

def detectLabel(imagePath):
    # data = sdl.loadImage(imagePath=imagePath)
    # data = data.astype("float") / 255.0
    # prediction = model_traffic_lights_nn.predict(data)
    # label = np.argmax(prediction)
    image = cv2.imread(imagePath)

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img_inception = cv2.resize(img_rgb, (224, 224))
    img_inception = np.array([preprocess_input(img_inception)])

    preds = model_traffic_lights_nn.predict(img_inception)

    label = np.argmax(preds)
    # score_light = str(int(np.max(preds) * 100))
    # print(classLabels[label], score_light)
    # Viết label lên ảnh
    # cv2.putText(image, "label: {}".format(classLabels[label] + score_light), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # # Hiển thị ảnh
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    return classLabels[label]