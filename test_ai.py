import tensorflow as tf
from tensorflow_addons.metrics import F1Score
import os
import numpy as np
from PIL import Image

try1 = tf.keras.models.load_model('model.h5', compile = True)

path = 'test_data/mussel/'

files = os.listdir(path)

xtest = [np.array(Image.open(path+f).convert('RGB').resize((150,150))) for f in files]
xtest = np.array(xtest).squeeze()

ypred = try1.predict(xtest, batch_size = 32, verbose =1)
label = ypred.argmax(1) 
print((label[label == 0].size)/label.size, (label[label == 1].size)/label.size)

print(label)



