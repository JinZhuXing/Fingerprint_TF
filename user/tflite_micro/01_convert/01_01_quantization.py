# %%
# import packages
import tensorflow as tf
import os


print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("GPU: ", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")



# %%
# load model
model_path = '../../../model/result/fp160.h5'
model_dir = os.path.dirname(model_path)
model = tf.keras.models.load_model(model_path)
model.summary()



# %%
# convert
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model = converter.convert()
open('../../../model/user/tflite_micro/fp160.tflite', 'wb').write(quantized_model)



# %%
