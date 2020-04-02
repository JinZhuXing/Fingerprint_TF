# %%
import numpy as np
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

import os

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("GPU: ", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")



# %%
# load data
img_data = np.load('../../dataset/np_data/img_train.npy')
label_data = np.load('../../dataset/np_data/label_train.npy')
img_real = np.load('../../dataset/np_data/img_real.npy')
label_real = np.load('../../dataset/np_data/label_real.npy')

print(img_data.shape, label_data.shape)
print(img_real.shape, label_real.shape)



# %%
# load model
model_path = '../../model/result/fp160.h5'
model_feature_path = '../../model/result/fp160_feature.h5'
feature_model = tf.keras.models.load_model(model_feature_path)
model = tf.keras.models.load_model(model_path)
feature_model.summary()
model.summary()



# %%
# match two image
input_idx = 0
db_idx = 1
input_img = img_real[input_idx].reshape((1, 160, 160, 1)).astype(np.float32) / 255.
db_img = img_real[db_idx].reshape((1, 160, 160, 1)).astype(np.float32) / 255.
pred_right = model.predict([input_img, db_img])

# show result
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title('Input: %s' %input_idx)
plt.imshow(input_img.squeeze(), cmap='gray')
plt.subplot(1, 2, 2)
if (pred_right > 0.9):
    plt.title('O: %.02f, %s' % (pred_right, db_idx))
else:
    plt.title('X: %.02f, %s' % (pred_right, db_idx))
plt.imshow(db_img.squeeze(), cmap='gray')



# %%
