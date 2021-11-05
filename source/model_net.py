from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model


# model network class
class Model_Net:
    def __init__(self, img_width, img_height) -> None:
        self.img_width = img_width
        self.img_height = img_height

    # model definition
    def feature_net(self, inputs):
        # Convolutional layers.
        conv_1 = keras.layers.Conv2D(filters=32,
                                    kernel_size=(3, 3),
                                    activation='relu')
        conv_2 = keras.layers.Conv2D(filters=64,
                                    kernel_size=(3, 3),
                                    strides=(1, 1),
                                    padding='valid',
                                    activation='relu')
        conv_3 = keras.layers.Conv2D(filters=64,
                                    kernel_size=(3, 3),
                                    strides=(1, 1),
                                    padding='valid',
                                    activation='relu')
        conv_4 = keras.layers.Conv2D(filters=64,
                                    kernel_size=(3, 3),
                                    strides=(1, 1),
                                    padding='valid',
                                    activation='relu')
        conv_5 = keras.layers.Conv2D(filters=64,
                                    kernel_size=[3, 3],
                                    strides=(1, 1),
                                    padding='valid',
                                    activation='relu')
        conv_6 = keras.layers.Conv2D(filters=128,
                                    kernel_size=(3, 3),
                                    strides=(1, 1),
                                    padding='valid',
                                    activation='relu')
        conv_7 = keras.layers.Conv2D(filters=128,
                                    kernel_size=[3, 3],
                                    strides=(1, 1),
                                    padding='valid',
                                    activation='relu')

        # Pooling layers.
        pool_1 = keras.layers.MaxPool2D(pool_size=(2, 2),
                                        strides=(2, 2),
                                        padding='valid')
        pool_2 = keras.layers.MaxPool2D(pool_size=(2, 2),
                                        strides=(2, 2),
                                        padding='valid')
        pool_3 = keras.layers.MaxPool2D(pool_size=(2, 2),
                                        strides=(2, 2),
                                        padding='valid')
        pool_4 = keras.layers.MaxPool2D(pool_size=[2, 2],
                                        strides=(1, 1),
                                        padding='valid')
        
        # Batch norm layers
        bn_1 = keras.layers.BatchNormalization()
        bn_2 = keras.layers.BatchNormalization()
        bn_3 = keras.layers.BatchNormalization()
        bn_4 = keras.layers.BatchNormalization()
        bn_5 = keras.layers.BatchNormalization()
        bn_6 = keras.layers.BatchNormalization()
        bn_7 = keras.layers.BatchNormalization()

        # |== Layer 1 ==|
        x = conv_1(inputs)
        x = bn_1(x)
        x = pool_1(x)

        # |== Layer 2 ==|
        x = conv_2(x)
        x = bn_2(x)
        x = conv_3(x)
        x = bn_3(x)
        x = pool_2(x)

        # |== Layer 3 ==|
        x = conv_4(x)
        x = bn_4(x)
        x = conv_5(x)
        x = bn_5(x)
        x = pool_3(x)

        # |== Layer 4 ==|
        x = conv_6(x)
        x = bn_6(x)
        x = conv_7(x)
        x = bn_7(x)
        outputs = pool_4(x)

        return outputs


    def build_model(self):
        # Conv layers.
        conv_8 = keras.layers.Conv2D(filters=256,
                                    kernel_size=[3, 3],
                                    strides=(1, 1),
                                    padding='valid',
                                    activation='relu')

        # Dense layers.
        dense_1 = keras.layers.Dense(units=1024,
                                    activation='relu',
                                    use_bias=True)
        dense_2 = keras.layers.Dense(units=1,
                                    activation='sigmoid',
                                    use_bias=True)

        # Batch norm layers.
        bn_8 = keras.layers.BatchNormalization()
        bn_9 = keras.layers.BatchNormalization()

        # Flatten layers.
        flatten_1 = keras.layers.Flatten()

        # All layers got. Define the forward propgation.
        input_1 = layers.Input(shape = (self.img_width, self.img_height, 1), name = 'image_input_1')
        input_2 = layers.Input(shape = (self.img_width, self.img_height, 1), name = 'image_input_2')
        output_1 = self.feature_net(input_1)
        output_2 = self.feature_net(input_2)

        sub = layers.Multiply()([output_1, output_2])

        # |== Layer 5 ==|
        x = conv_8(sub)
        x = bn_8(x)

        # |== Layer 6 ==|
        x = flatten_1(x)
        x = dense_1(x)
        x = bn_9(x)
        out_res = dense_2(x)

        model = Model(inputs = [input_1, input_2], outputs = out_res)

        # show summary
        print('Main Model Summary')
        model.summary()

        # compile
        model.compile(loss = 'binary_crossentropy', optimizer = optimizers.RMSprop(learning_rate=1e-4), metrics = ['acc'])

        return model
