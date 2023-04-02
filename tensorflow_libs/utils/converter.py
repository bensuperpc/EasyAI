import cv2 as cv
import tensorflow as tf
import numpy as np
from loguru import logger


def image_to_tensor(img=None, _input_image_res=(256, 256), dtype=tf.float16):

    if img is None:
        logger.error("Image is None !")
        return

    # Correct image format
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, _input_image_res)

    # Convert to tensor
    img = tf.convert_to_tensor(img, dtype=dtype)
    img = tf.expand_dims(img, 0)
    return img

def predict_v1(img_path=None, _model=None, _input_image_res=(256, 256), _class_names=None):
    if img_path is None:
        logger.error("Image path is None !")
        return

    image = cv.imread(img_path, 0)

    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    image = cv.resize(image, _input_image_res)

    #image = np.asarray(image).reshape(-1, self._img_height, self._img_width, 3)

    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.expand_dims(image, 0)

    if _model is None:
        logger.error("Model is None !")
        return

    predictions = np.argmax(_model.predict(image, use_multiprocessing=True))

    #logger.debug(f"Predictions: {predictions}")
    #logger.debug(f"Predictions: {self._class_names[predictions]} ")
    return predictions

def predict_v2(img_path=None, _model=None, _input_image_res=(256, 256), _class_names=None):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=_input_image_res)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    if _model is None:
        logger.error("Model is None !")
        return

    predictions = np.argmax(_model.predict(img, use_multiprocessing=True))

    #score = tf.nn.softmax(predictions[0])
    #logger.debug(f"Predictions: {self._class_names[predictions]}")
    return predictions