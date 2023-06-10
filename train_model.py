import collections  # Handles specialized container datatypes
import os

import cv2  # Computer vision library
import matplotlib.pyplot as plt  # Plotting library
import numpy as np  # Scientific computing library
from sklearn.metrics import classification_report

import object_detection  # Custom object detection program
import sys
import tensorflow as tf  # Machine learning library
from tensorflow import keras  # Library for neural networks
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam, Adadelta
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

from Model.InceptionV3 import InceptionV3Custom
from Utils.draw_plot import  plot_trend_by_epoch, show_history

sys.path.append('../')

# Show the version of TensorFlow and Keras that I am using
print("TensorFlow", tf.__version__)
print("Keras", keras.__version__)

# def Transfer(n_classes, freeze_layers=True):
#     """
#     Use the InceptionV3 neural network architecture to perform transfer learning.
#
#     :param:n_classes Number of classes
#     :param:freeze_layers If True, the network's parameters don't change.
#     :return The best neural network
#     """
#     print("Loading Inception V3...")
#
#     base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#
#     print("Inception V3 has finished loading.")
#
#     # Display the base network architecture
#     print('Layers: ', len(base_model.layers))
#     print("Shape:", base_model.output_shape[1:])
#     print("Shape:", base_model.output_shape)
#     print("Shape:", base_model.outputs)
#     base_model.summary()
#
#     # Create the neural network. This network uses the Sequential
#     # architecture where each layer has one
#     # input tensor (e.g. vector, matrix, etc.) and one output tensor
#     top_model = Sequential()
#
#     # Our classifier model will build on top of the base model
#     top_model.add(base_model)
#     top_model.add(GlobalAveragePooling2D())
#     top_model.add(Dropout(0.5))
#     top_model.add(Dense(1024, activation='relu'))
#     top_model.add(BatchNormalization())
#     top_model.add(Dropout(0.5))
#     top_model.add(Dense(512, activation='relu'))
#     top_model.add(Dropout(0.5))
#     top_model.add(Dense(128, activation='relu'))
#     top_model.add(Dense(n_classes, activation='softmax'))
#
#     # Freeze layers in the model so that they cannot be trained (i.e. the
#     # parameters in the neural network will not change)
#     # No Finetune
#     if freeze_layers:
#         for layer in base_model.layers:
#             layer.trainable = False
#
#     return top_model


train_datagen = ImageDataGenerator(rescale=1. / 255)

val_datagen = ImageDataGenerator(rescale=1. / 255)

test_datagen = ImageDataGenerator(rescale=1. / 255)

shape = (224, 224)

train_it = train_datagen.flow_from_directory(directory="traffic_light_dataset/train", class_mode="categorical",
                                             shuffle=True,
                                             batch_size=64,
                                             color_mode="rgb",
                                             # subset='training',
                                             seed=16
                                             )
val_it = val_datagen.flow_from_directory(directory="traffic_light_dataset/val", class_mode="categorical",
                                         batch_size=64,
                                         shuffle=True,
                                         color_mode="rgb",
                                         # subset='validation',
                                         seed=16
                                         )

test_it = val_datagen.flow_from_directory(directory="traffic_light_dataset/test",
                                          batch_size=1,
                                          shuffle=False,
                                          seed=42,
                                          color_mode="rgb")

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_it.classes), y=train_it.classes)
print("CLASS WEIGHT: ", class_weights)
train_class_weights = dict(enumerate(class_weights))
print("TRAIN CLASS WEIGHT: ", train_class_weights)

# print(train_class_weights)

# Save the best model as traffic.h5
checkpoint = ModelCheckpoint("traffic.h5", monitor='val_loss', mode='min', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(min_delta=0.0005, patience=10, verbose=1)

# Generate model using transfer learning
model = InceptionV3Custom.build(n_classes=4, freeze_layers=True)

# Display a summary of the neural network model
model.summary()

# Configure the model parameters for training
model.compile(loss=categorical_crossentropy, optimizer=Adadelta(
    learning_rate=1.0, rho=0.95, epsilon=1e-08, decay=1.0 / 60), metrics=['accuracy'])

history_object = model.fit(train_it, steps_per_epoch=train_it.n // train_it.batch_size,
                           callbacks=[checkpoint, early_stopping],
                           validation_data=val_it, validation_steps=val_it.n // val_it.batch_size, epochs=60, )

# Get the loss value and metrics values on the test data set
# score = model.evaluate(x_valid, y_valid, verbose=0)
score = model.evaluate(val_it, verbose=0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])

print("[INFO] evaluating network...")
n_test_steps = test_it.n // test_it.batch_size
test_it.reset()
y_pred = model.predict(test_it, steps=n_test_steps, verbose=1)
y_pred = np.argmax(y_pred, axis=1)
print(classification_report(test_it.classes, y_pred, target_names=["green", "off", "red", "yellow"]))

# Show/save plot
tr_accuracy, val_accuracy = history_object.history["accuracy"], history_object.history["val_accuracy"]
plot_trend_by_epoch(tr_accuracy, val_accuracy, "Model Accuracy", "Accuracy", "plot_accu.png")
plt.clf()
tr_loss, val_loss = history_object.history["loss"], history_object.history["val_loss"]
plot_trend_by_epoch(tr_loss, val_loss, "Model Loss", "Loss", "plot_loss.png")


show_history(history_object)
