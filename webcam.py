import glob

import cv2
from datetime import date
from datetime import datetime

import numpy as np
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input, decode_predictions

fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 0.5
fontcolor = (0, 255, 0)
fontcolor1 = (0, 0, 255)

# Khởi tạo danh sách nhãn
classLabels = ["green", "off", "red", "yellow"]

print("[INFO] Nạp model mạng pre-trained ...")
model = load_model("traffic.h5")

# # Khởi tạo camera
# cam = cv2.VideoCapture(0)
#
# while (True):
#
#     # Đọc ảnh từ camera
#     ret, img = cam.read()
#
#     # Lật ảnh cho đỡ bị ngược
#     img = cv2.flip(img, 1)
#
#     frame1 = cv2.resize(img,(224,224))
#     frame1_tensor = frame1.reshape(1, 224, 224, 3)
#
#     preds = model.predict(frame1_tensor)
#
#     label = np.argmax(preds)
#
#     score_light = str(int(np.max(preds) * 100))
#
#     cv2.putText(img, "label: {}".format(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#
#     cv2.imshow('frame', img)
#
#     # Nếu nhấn q thì thoát
#     if cv2.waitKey(1) == ord('q'):
#         break
# cam.release()
# cv2.destroyAllWindows()
# load the pre-trained network
print("[INFO] Nạp model mạng pre-trained ...")

model = load_model("traffic.h5")

DIR = 'test/*'

for image_path in glob.glob(DIR):

    # image = cv2.imread("test/y333.jpg")
    image = cv2.imread(image_path)

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img_inception = cv2.resize(img_rgb, (224, 224))
    img_inception = np.array([preprocess_input(img_inception)])

    preds = model.predict(img_inception)

    label = np.argmax(preds)

    score_light = str(int(np.max(preds) * 100))

    print(classLabels[label], score_light)

    # Viết label lên ảnh
    cv2.putText(image, "label: {}".format(classLabels[label] + score_light), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Hiển thị ảnh
    cv2.imshow("Image", image)
    cv2.waitKey(0)
