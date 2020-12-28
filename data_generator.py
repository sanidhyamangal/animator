"""
author: Sanidhya Mangal
github: sanidhyamangal
"""
import os  # for os related ops

import tensorflow as tf  # for deep learning based ops

from models import AnimeGenerator  # generative models
from utils import \
    generate_and_save_images  # function to generate and save images


class AnimeImageGenerator:
    """
    Class for generating animes images
    """
    def __init__(self,
                 path_to_checkpoint: str = './training_checkpoints',
                 *args,
                 **kwargs):
        self.generator = AnimeGenerator()
        # creation of checkpoint dirs

        checkpoint = tf.train.Checkpoint(generator=self.generator)

        checkpoint.restore(tf.train.latest_checkpoint(path_to_checkpoint))

    def generate_images(self, image_name: str = "generated.png"):
        seed = tf.random.normal([16, 100])
        generate_and_save_images(self.generator, image_name, seed, False)


gen = AnimeImageGenerator()
for i in range(1, 51):
    gen.generate_images("generated{}.png".format(i))
