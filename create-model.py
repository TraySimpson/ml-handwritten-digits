import os
import cv2                          # Computer Vision, used to read in image data
import numpy as np                  # Numpy arrays
import matplotlib.pyplot as plt     # Visualization of digits
import tensorflow as tf             # ML stuff

mnist = tf.keras.datasets.mnist     # load the dataset
# split data into training and testing data
# x data is the image data itself, y data is the label (ex. "2")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize the data from 0-255 values to 0-1
# TODO research keras, and axis param
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# TODO research different model types
model = tf.keras.models.Sequential()
# Add "flattened" data as layer to our ML model
# "flattened" just converts the 2D 28x28 data into a 1D 1x728 data set
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# TODO research activation function
model.add(tf.keras.layers.Dense(128, activation='relu'))
# Output layer
# Softmax AF ensure that all values add up to one (probability)
# this way each neuron provides a "confidence" value for output
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# TODO no explanation provided on any params :) great vid
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model, epoch is the number of times model sees the same data
# TODO verify
model.fit(x_train, y_train, epochs=10)

model.save('handwritten.model')