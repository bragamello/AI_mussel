from tensorflow_addons.metrics import F1Score

import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt

path_mussel = 'mussel/'
path_nomussel = 'nomussel/'

files_mussel = os.listdir(path_mussel)
files_nomussel = os.listdir(path_nomussel)

from PIL import Image

x_mussel_base = [np.array(Image.open(path_mussel+f).convert('RGB').resize((150,150))) for f in files_mussel]
x_nomussel_base = [np.array(Image.open(path_nomussel+f).convert('RGB').resize((150,150))) for f in files_nomussel]

x_mussel = np.array(x_mussel_base).squeeze()
x_nomussel = np.array(x_nomussel_base).squeeze()

x_train = np.vstack([x_mussel,x_nomussel])
y_train = np.vstack([0*np.ones((len(x_mussel),1)),1*np.ones((len(x_nomussel),1))])

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train,num_classes=2)

from tensorflow import keras

input_shape = (150, 150, 3)
inputs = tf.keras.Input(shape=input_shape, name = 'model_inputs')

x = keras.layers.experimental.preprocessing.RandomZoom(0.3)(inputs)
x = keras.layers.experimental.preprocessing.RandomFlip(mode = 'horizontal')(x)
x = keras.layers.experimental.preprocessing.RandomRotation(0.028)(x)
x = keras.layers.experimental.preprocessing.Rescaling(1./255)(x)

x = keras.layers.Conv2D(32, 3, activation='relu', input_shape = input_shape)(x)
x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
x = keras.layers.Conv2D(64, 3, activation='relu')(x)
x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
x = keras.layers.Conv2D(128, 3, activation='relu')(x)
x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
x = keras.layers.Conv2D(256, 3, activation='relu')(x)
x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
x = keras.layers.Dense(256, activation = 'relu',name = 'dense_before_final')(x)
x = keras.layers.Dropout(0.5, name = 'drop_out')(x)
x = keras.layers.GlobalAveragePooling2D(name='globalavgpool')(x)
out = keras.layers.Dense(2, activation = 'sigmoid',name = 'output_layer')(x)
model = keras.Model(inputs, out)

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

checkpointer = ModelCheckpoint(filepath='model.h5',verbose=1, save_best_only = True,monitor = 'val_f1_score', mode = 'max')

csv_logger = CSVLogger(filename='model.csv')

callbacks_list = [checkpointer, csv_logger]

from sklearn.utils.class_weight import compute_class_weight
cw = compute_class_weight('balanced', classes = np.unique(y_train.argmax(1)), y = y_train.argmax(1))
class_weight = dict(zip([0,1],cw))

from sklearn.model_selection import train_test_split
xtra, xva, ytra, yva = train_test_split(x_train, y_train, test_size=0.15)

model.compile(optimizer=tf.keras.optimizers.Adam(),loss = tf.keras.losses.BinaryCrossentropy(),metrics=[F1Score(num_classes=2,average='macro'),'acc'])

history = model.fit(xtra, ytra, epochs = 100, callbacks = callbacks_list, validation_data=(xva,yva), batch_size=16, shuffle=True, class_weight=class_weight)

model.save_weights('modelweights.h5')
