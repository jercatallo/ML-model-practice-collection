#NOTE: https://github.com/tensorflow/tensorflow/issues/9829


import pickle
import time
# import tenserf as tfjs
import tensorflowjs as tfjs
import os
import tensorflow as tf
# os.add_dll_directory()

# TO USE CPU In training
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    

# if tf.test.gpu_device_name():
#     print('GPU found')
# else:
#     print("No GPU found")
    
# Read file using pickle
# rb - read in binary
X = pickle.load(open('X.pkl', 'rb'))
y = pickle.load(open('y.pkl', 'rb'))


# feature scaling
# get all the values and dividing them into 255
# less than the values faster calculation
X = X/255

# 16 IMAGES, 100 HEIGHT, 100 WIDTH, 3 CHANNELS(RGB)
X.shape

from tensorflow.python.keras.engine.sequential import Sequential
# Conv2D used for Convolution, MaxPooling2D used for max pooling, Flatten use for flattening, Dense is the hidden layer for the neural network
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import TensorBoard
NAME = f'cat-vs-dog-prediction-{int(time.time())}'

tenserboard = TensorBoard(log_dir=f'logs\\{NAME}\\')


model = Sequential()

# This is the convolution layer
# 64 Convolution layers, Feature detector 3x3 Metrics, then we will do activation
model.add(Conv2D(64,(3,3), activation="relu"))
# Max pooling layer
model.add(MaxPooling2D((2,2)))

# HERE WE WILL REPEAT THE CONVOLUTIONAL LAYER SO THAT IT CAN HAVE 2 LAYER, MORE ACCURACY AND LOWER LOSS

# This is the convolution layer
# 64 Convolution layers, Feature detector 3x3 Metrics, then we will do activation
model.add(Conv2D(64,(3,3), activation="relu"))
# Max pooling layer
model.add(MaxPooling2D((2,2)))

# HERE WE WILL REPEAT THE CONVOLUTIONAL LAYER SO THAT IT CAN HAVE 3 LAYER, MORE ACCURACY AND LOWER LOSS

# This is the convolution layer
# 64 Convolution layers, Feature detector 3x3 Metrics, then we will do activation
model.add(Conv2D(64,(3,3), activation="relu"))
# Max pooling layer
model.add(MaxPooling2D((2,2)))

# Flattening
model.add(Flatten())

# For neural network
# 128 neurons, input_shape = 100, 100, 3
model.add(Dense(128, input_shape= X.shape[1:], activation='relu'))

# no input_shape because it wil ldirectly get the shape of the previous layer
model.add(Dense(128, activation='relu'))

# Output layer, 2 neurons because 2 label(cat,dog), softmax and sigmoid return value between 0 and 1
model.add(Dense(2, activation= 'softmax'))

# Loss and optimizers
model.compile(optimizer= 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# validation_split will split the datasets to training and assigning for validation(testing)
model.fit(X,y, epochs=9, validation_split=0.1, batch_size=32, callbacks=[tenserboard])

tfjs.converters.save_keras_model(model,'models')