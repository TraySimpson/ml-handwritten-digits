import os
import cv2                          # Computer Vision, used to read in image data
import numpy as np                  # Numpy arrays
import matplotlib.pyplot as plt     # Visualization of digits
import tensorflow as tf             # ML stuff

mnist = tf.keras.datasets.mnist     # load the dataset
# split data into training and testing data
# x data is the image data itself, y data is the label (ex. "2")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# loads the saved model
model = tf.keras.models.load_model('handwritten.model')

loss, accuracy = model.evaluate(x_test, y_test)

image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This digit is probably {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error, this guy sucks!")
    finally:
        image_number += 1

# print(loss)
# print(accuracy)