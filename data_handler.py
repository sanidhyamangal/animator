"""
Author: Sanidhya Mangal
github: sanidhyamangal
"""

import tensorflow as tf  # for deep learning purpose
import pathlib  # for path ops
import matplotlib.pyplot as plt


def process_image(file_path: str) -> tf.Tensor:
    raw_image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(raw_image)
    image = tf.image.resize(image, size=(64, 64))

    return (image - 127.0) / 127.0


def get_anime_dataset(path: str, batch_size: int = 64) -> tf.data.Dataset:
    path_to_images = pathlib.Path(path)
    images = [str(image) for image in path_to_images.glob("*.jpg")]
    return tf.data.Dataset.from_tensor_slices(
        (images)).map(process_image,
                      num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(
                          batch_size).prefetch(tf.data.experimental.AUTOTUNE)
