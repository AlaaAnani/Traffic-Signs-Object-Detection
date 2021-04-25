# %%
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
from skimage.io import imread 
from skimage.transform import rescale, resize, downscale_local_mean

img_height = 100
img_width = 100

from tensorflow.keras import layers
X, y = [], []
base0 = 'dataset/0/'
base1 = 'dataset/1/'
for path in os.listdir(base0):
    img = imread(base0+path)
    img = resize(img, (img_width, img_height))
    img= img/255.0
    X.append(img)
    y.append(0)
for path in os.listdir(base1):
    img = imread(base1+path)
    img = resize(img, (img_width, img_height))
    img= img/255.0
    X.append(img)
    y.append(1)

X = np.array(X)
y = np.array(y)

# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import *
model = Sequential()
model.add(Conv2D(32, 3, input_shape=X[0].shape, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32, 3, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32, 3, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, 3, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile(
  optimizer='adam',
  loss='binary_crossentropy',
  metrics=['accuracy'])
model.summary()

model.fit(X, y, batch_size=64, epochs=260)

model.save("cnn_model_3")
