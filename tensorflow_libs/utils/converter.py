import cv2 as cv
import tensorflow as tf
import numpy as np
from loguru import logger


def image_to_tensor(img=None, input_shape=(256, 256), dtype=tf.float16):
    # Correct image format
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, input_shape)

    # Convert to tensor
    img = tf.convert_to_tensor(img, dtype=dtype)
    img = tf.expand_dims(img, 0)
    return img
