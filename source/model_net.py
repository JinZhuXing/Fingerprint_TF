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
        conv_1 = keras.layers.Conv2D(filters=16,
                                    kernel_size=(3, 3),
                                    activation='relu')
        conv_2 = keras.layers.Conv2D(filters=32,
                                    kernel_size=(3, 3),
                                    strides=(1, 1),
                                    padding='valid',
                                    activation='relu')
        conv_3 = keras.layers.Conv2D(filters=64,
                                    kernel_size=(3, 3),
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

        # |== Layer 1 ==|
        x = conv_1(inputs)
        x = pool_1(x)

        # |== Layer 2 ==|
        x = conv_2(x)
        x = pool_2(x)

        # |== Layer 3 ==|
        x = conv_3(x)
        x = pool_3(x)

        # |== Layer 4 ==|
        outputs = pool_4(x)

        return outputs


    def build_model(self):
        # Dense layers.
        dense_1 = keras.layers.Dense(units=1024,
                                    activation='relu',
                                    use_bias=True)
        dense_2 = keras.layers.Dense(units=1,
                                    activation='sigmoid',
                                    use_bias=True)

        # Flatten layers.
        flatten_1 = keras.layers.Flatten()

        # All layers got. Define the forward propgation.
        input_1 = layers.Input(shape = (self.img_width, self.img_height, 1), name = 'image_input_1')
        input_2 = layers.Input(shape = (self.img_width, self.img_height, 1), name = 'image_input_2')
        output_1 = self.feature_net(input_1)
        output_2 = self.feature_net(input_2)

        sub = layers.Subtract()([output_1, output_2])

        # |== Layer 5 ==|
        x = flatten_1(sub)
        x = dense_1(x)
        out_res = dense_2(x)

        model = Model(inputs = [input_1, input_2], outputs = out_res)

        # show summary
        print('Main Model Summary')
        model.summary()

        # compile
        model.compile(loss = 'binary_crossentropy', optimizer = optimizers.RMSprop(learning_rate=1e-4), metrics = ['acc'])

        return model
