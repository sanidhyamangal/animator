"""
author: Sanidhya Mangal
github: sanidhyamangal
"""

from functools import partial  # for creation of partial functions

import tensorflow as tf  # for deep learning based stuff

# partial functions for the ease of development
Conv2DT = partial(tf.keras.layers.Conv2DTranspose,
                  kernel_size=(5, 5),
                  strides=(2, 2),
                  padding="same",
                  use_bias=False)
Conv2D = partial(tf.keras.layers.Conv2D,
                 kernel_size=(5, 5),
                 strides=(2, 2),
                 padding="same")


class AnimeGenerator(tf.keras.models.Model):
    """
    A generative model for anine generator
    """
    def __init__(self, *args, **kwargs):
        super(AnimeGenerator, self).__init__(*args, **kwargs)

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(8 * 8 * 512,
                                  use_bias=False,
                                  input_shape=(100, )),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Reshape((8, 8, 512)),
            Conv2DT(filters=256, strides=(1, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            Conv2DT(filters=256),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            Conv2DT(filters=128),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            Conv2DT(filters=3, activation=tf.nn.tanh)
        ])

    def call(self, inputs, *args, **kwargs):
        return self.model(inputs)


class AnimeDiscriminator(tf.keras.models.Model):
    """
    A discriminative model for pokemon discriminator
    """
    def __init__(self, *args, **kwargs):
        super(AnimeDiscriminator, self).__init__(*args, **kwargs)

        self.model = tf.keras.models.Sequential([
            Conv2D(filters=64),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(0.3),
            Conv2D(filters=128),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(0.3),
            Conv2D(filters=256),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(0.3),
            Conv2D(filters=512),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1)
        ])

    def call(self, inputs, *args, **kwargs):
        return self.model(inputs)
