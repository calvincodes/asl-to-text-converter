"""
@author: arun.jose
@author: arpit.jain
"""

import numpy as np
import cv2
import operator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
K.set_image_dim_ordering('th')

# Input image dimensions
img_rows, img_cols = 100, 100

# Number of channels used (as we are binarizing the images, this is 1)
img_channels = 1

# Batch_size to train
batch_size = 32

# Number of output classes
nb_classes = 5

# Number of epochs to train
nb_epoch = 15

# Total number of convolutional filters to use
nb_filters = 32
# Maximum pooling
nb_pool = 2
# Size of convolution kernel
nb_conv = 3

class_labels = ["0", "1", "2", "3", "4"]

label_array = {}


# This method is used for generation of histogram after classifying the gesture
def display_histogram(plot):
    global label_array
    h = 450
    w = 45
    x = 30
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Vertical BOTTOM to TOP histograms
    for items in label_array:
        mul = (label_array[items]) / 100
        cv2.line(plot, (x, 510), (x, 490 - int(h * mul)), (255, 178, 102), w)
        cv2.putText(plot, items, (x-5, 500), font, 0.9, (0, 255, 255), 2, 3)
        x = x + w + 30

    return plot


# Load Convolutional neural network model
def load_conv_neural_net():
    cnn_model = Sequential()

    cnn_model.add(
        Conv2D(nb_filters, (nb_conv, nb_conv), padding='valid', input_shape=(img_channels, img_rows, img_cols)))
    convolutional_1 = Activation('relu')
    cnn_model.add(convolutional_1)
    cnn_model.add(Conv2D(nb_filters, (nb_conv, nb_conv)))
    convolutional_2 = Activation('relu')
    cnn_model.add(convolutional_2)
    cnn_model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    cnn_model.add(Dropout(0.5))

    cnn_model.add(Flatten())
    cnn_model.add(Dense(128))
    cnn_model.add(Activation('relu'))
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Dense(nb_classes))
    cnn_model.add(Activation('softmax'))

    cnn_model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    # Load pre-trained weights from the hdf5 file
    file_name = "weights_01234_100x100_10epochs_99accuracy.hdf5"
    print("Loading weights from ", file_name)
    cnn_model.load_weights(file_name)

    layer = cnn_model.layers[11]
    K.function([cnn_model.layers[0].input, K.learning_phase()], [layer.output, ])

    return cnn_model


def classify_asl_symbol(model, img):
    global class_labels, label_array

    image = np.array(img).flatten()
    # Reshape the loaded image to match training image dimensions
    image = image.reshape(img_channels, img_rows, img_cols)
    image = image.astype('float32')
    image = image / 255 # Normalization
    # Reshape the loaded image to match training image dimensions
    rimage = image.reshape(1, img_channels, img_rows, img_cols)

    probability_array = model.predict_proba(rimage)

    dictionary = {}
    i = 0
    for items in class_labels:
        # Rescale probabilities on a scale of 100.
        dictionary[items] = probability_array[0][i] * 100
        i = i + 1

    classified_index = max(dictionary.items(), key=operator.itemgetter(1))[0]
    probability = dictionary[classified_index]

    if probability > 60.0:
        label_array = dictionary
        return class_labels.index(classified_index)
    else:
        return 1


