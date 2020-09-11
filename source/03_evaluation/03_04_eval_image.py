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
# load model
model_path = '../../model/result/fp160.h5'
model_feature_path = '../../model/result/fp160_feature.h5'
feature_model = tf.keras.models.load_model(model_feature_path)
model = tf.keras.models.load_model(model_path)
feature_model.summary()
model.summary()



# %%
input_idx = '00000'
db_idx = '00001'
input_file = '../../dataset/real_data/' + input_idx + '.bmp'
db_file = '../../dataset/real_data/' + db_idx + '.bmp'

input_img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
input_img = input_img.reshape((1, 160, 160, 1)).astype(np.float32) / 255.
db_img = cv2.imread(db_file, cv2.IMREAD_GRAYSCALE)
db_img = db_img.reshape((1, 160, 160, 1)).astype(np.float32) / 255.
pred_right = model.predict([input_img, db_img])

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title('Input: %s' %input_idx)
plt.imshow(input_img.squeeze(), cmap='gray')
plt.subplot(1, 2, 2)
if (pred_right > 0.5):
    plt.title('O: %.02f, %s' % (pred_right, db_idx))
else:
    plt.title('X: %.02f, %s' % (pred_right, db_idx))
plt.imshow(db_img.squeeze(), cmap='gray')



# %%
