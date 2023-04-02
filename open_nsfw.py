#!/usr/bin/env python

import argparse
from argparse import ArgumentParser

import os
import pathlib
import datetime
import sys
from pathlib import Path
import time
import json

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import loguru
from loguru import logger
import cv2 as cv

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(2)

from tensorflow_libs import ben_model_v1, ci_model_v1, vgg11, vgg13, vgg16, vgg19, ben_model_v1_class
from tensorflow_libs import display_history, plot_image, plot_value_array, display_predict
from tensorflow_libs import gpu

class AI:
    __author__ = "Bensuperpc"
    __copyright__ = None
    __credits__ = ["None", "None"]
    __license__ = "MIT"
    __version__ = "1.0.0"
    __maintainer__ = "Bensuperpc"
    __email__ = "bensuperpc@gmail.com"
    __status__ = "Development"
    __compatibility__ = ["Linux", "Windows", "Darwin"]
    __name__ = "AI"

    def gpu(self):
        gpu()

    # Get label from file path (Tensor type) and list of class true/false
    def get_label(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        return parts[-2] == self._class_names
        
    
        #one_hot = parts[-2] == self._class_names
        #tf.print(parts[-2], output_stream=sys.stdout)
        #tf.print(self._class_names, output_stream=sys.stdout)
        #tf.print(one_hot, output_stream=sys.stdout)
        #tf.print(tf.argmax(one_hot), output_stream=sys.stdout)
        #return tf.argmax(one_hot) # and return the index of the label
    
    def decode_img(self, file_path):
        img = tf.io.read_file(file_path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)

        return tf.image.resize(img, [self._img_height, self._img_width])

    def process_path(self, file_path):
        label = self.get_label(file_path)
        img = self.decode_img(file_path)

        return img, label

    def configure_for_performance(self, ds):
        if self._data_augmentation:
            logger.debug("Using data augmentation")
            data_augmentation = tf.keras.Sequential([
                layers.RandomFlip("horizontal_and_vertical"),
                layers.RandomRotation(0.2),
                layers.RandomZoom(0.1),
                # layers.RandomContrast(0.1),
                # layers.RandomBrightness(0.1),
            ])
            logger.debug(f"Dataset size without augmentation: {len(ds)}")
            ds_aug = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                            num_parallel_calls=self._AUTOTUNE)
            ds = ds.concatenate(ds_aug)
            logger.debug(f"Dataset size with augmentation: {len(ds)}")
            logger.debug("Data augmentation done")
        
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=1000)
        #ds = ds.repeat()
        ds = ds.batch(self._batch_size)
        ds = ds.prefetch(buffer_size=self._AUTOTUNE)
        return ds

    def get_model(self):
        # data_augmentation = tf.keras.Sequential([
        #  layers.RandomFlip("horizontal_and_vertical"),
        #  layers.RandomRotation(0.2),
        # ])

        model = ben_model_v1_class(input_shape=(self._img_height, self._img_width, 3), 
            output_size=len(self._class_names))

        model.build()
        model.summary()

        return model

    def get_ci_model(self):
        model = ci_model_v1(input_shape=(self._img_height, self._img_width, 3))

        # Add last layer
        if len(self._class_names) >= 2:
            model.add(layers.Dense(len(self._class_names), activation='softmax'))
        else:
            model.add(layers.Dense(len(self._class_names), activation='sigmoid'))

        model.build((None, self._img_height, self._img_width, 3))
        model.summary()

        return model

    def compile(self):
        self._model.compile(
            optimizer=self._optimizer,
            loss=self._loss,
            metrics=self._metrics)

    def load_model(self, model_path=None):
        # Load model
        if model_path is not None:
            self._model = keras.models.load_model(model_path)
            logger.debug(f"Model {model_path} loaded")
        elif model_path is None and self._model is not None:
            logger.debug("Model is already loaded")
        else:
            logger.warning("Model is None, load default model")
            self._model = self.get_model()

            # Remove last layer
            self._model.pop()
            self._model.add(layers.Dense(len(self._class_names)))


    def save_model(self, model_path=None):
        if model_path is not None:
            self._model.save(model_path, save_format='h5')
            logger.debug(f"Model saved to {model_path}")
        else:
            save_path = f"model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
            logger.warning(
                f"Model path is None, model saved to default path: {save_path}")

            self._model.save(save_path, save_format='h5')
            logger.debug(f"Model saved to: {save_path}")

    def load_weights(self, weights_path=None):
        if weights_path is not None:
            self._model.load_weights(weights_path)
            logger.debug(f"Weights {weights_path} loaded")
        elif weights_path is None and self._model is not None:
            logger.debug("Weights is already loaded")
        else:
            logger.warning("Weights is None, load default weights")
            self._model.load_weights("weights.h5")
            logger.debug("Weights loaded to: weights.h5")

    def save_weights(self, weights_path=None):
        if weights_path is not None:
            self._model.save_weights(weights_path)
            logger.debug(f"Weights saved to {weights_path}")
        else:
            save_path = f"weights_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
            logger.warning(
                "Weights path is None, weights saved to default path: {save_path}")
            self._model.save_weights(save_path)
            logger.debug("Weights saved to: {save_path}")

    def load_data(self, data_dir=None):
        if data_dir is not None:
            self._data_dir = data_dir

        if not isinstance(self._data_dir, pathlib.PurePath):
            self._data_dir = pathlib.Path(self._data_dir)

        logger.debug(f"Data loaded from {self._data_dir}")

        self._data_dir = pathlib.Path(self._data_dir)
        self._list_ds = tf.data.Dataset.list_files(str(self._data_dir / '*/*'))

        if not self._class_names:
            self._class_names = np.array(sorted(
                [item.name for item in self._data_dir.glob('*') if item.name != "LICENSE.txt"]))

        logger.info(f"Class names: {self._class_names}")
        logger.debug(f"Number of classes: {len(self._class_names)}")

        logger.debug(
            f"Number of images: {len(list(self._data_dir.glob('*/*.jpg')))}")

    def prepare_train(self):
        self._list_ds = tf.data.Dataset.list_files(str(self._data_dir/'*/*'))
        self._labeled_ds = self._list_ds.map(
            self.process_path, num_parallel_calls=self._AUTOTUNE)

        train_size = int(self._train_pourcent * len(self._labeled_ds))
        val_size = int(self._val_pourcent * len(self._labeled_ds))
        test_size = int(self._test_pourcent * len(self._labeled_ds))

        self._train_ds = self._labeled_ds.take(train_size)

        self._val_ds = self._labeled_ds.skip(train_size)
        self._val_ds = self._val_ds.take(val_size)

        self._test_ds = self._labeled_ds.skip(train_size + val_size)
        self._test_ds = self._test_ds.take(test_size)

        logger.debug(f"Train size: {len(self._train_ds)}")
        logger.debug(f"Val size: {len(self._val_ds)}")
        logger.debug(f"Test size: {len(self._test_ds)}")

        self._train_ds = self.configure_for_performance(self._train_ds)
        self._val_ds = self.configure_for_performance(self._val_ds)
        self._test_ds = self.configure_for_performance(self._test_ds)

    def train(self):
        logger.debug("Start training")
        logger.debug(f"Epochs: {self._epochs}")
        logger.debug(f"Batch size: {self._batch_size}")

        if self._tf_callbacks:
            logger.debug("Enable Tensorflow callback")
            self._history = self._model.fit(
                self._train_ds,
                epochs=self._epochs,
                validation_data=self._val_ds,
                callbacks=self._tf_callbacks,
                verbose=1
            )
        else:
            logger.debug("Tensorboard is disabled")
            self._history = self._model.fit(
                self._train_ds,
                epochs=self._epochs,
                validation_data=self._val_ds,
                verbose=1
            )

    def evaluate(self):
        logger.debug("Start evaluation")
        loss, accuracy = self._model.evaluate(self._test_ds)

        logger.debug(f"Loss: {loss * 100} %")
        logger.debug(f"Accuracy: {accuracy * 100} %")

        return loss, accuracy

    def predict_v1(self, img_path=None):
        logger.debug("Start prediction")
        if img_path is None:
            logger.warning("Image path is None !")
            return

        image = cv.imread(img_path, 0)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image = cv.resize(image, (self._img_height, self._img_width))

        #image = np.asarray(image).reshape(-1, self._img_height, self._img_width, 3)

        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.expand_dims(image, 0)

        if self._model is None:
            logger.warning("Model is None !")
            return

        predictions = np.argmax(self._model.predict(
            image, use_multiprocessing=True))

        #logger.debug(f"Predictions: {predictions}")
        #logger.debug(f"Predictions: {self._class_names[predictions]} ")
        return predictions

    def predict_v2(self, img_path=None):
        img = keras.preprocessing.image.load_img(
            img_path, target_size=(self._img_width, self._img_height))
        img = keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)

        if self._model is None:
            logger.warning("Model is None !")
            return

        predictions = np.argmax(self._model.predict(
            img, use_multiprocessing=True))
        #score = tf.nn.softmax(predictions[0])
        #logger.debug(f"Predictions: {self._class_names[predictions]}")
        return predictions

    def predict(self, img_path=None):
        logger.debug(f"Prediction path: {img_path}")

        if self._model is None:
            logger.error("Model is None !")
            return

        if img_path is None:
            logger.error("Image path is None !")
            return

        img_path = pathlib.Path(img_path)
        images = img_path.glob('*')

        for image in images:
            #start = time.process_time()
            prediction = self.predict_v2(str(image))
            #logger.warning(time.process_time() - start)
            logger.debug(
                f"Predictions: {self._class_names[prediction]} for {image}")

    def display_history(self):
        display_history(self._history, self._epochs)

    def display_predict(self):
        display_predict(self._model, self._test_ds, self._class_names, num_rows=5, num_cols=3)

    # Defining __init__ method
    def __init__(self, **kwargs):
        logger.info(f"Python version: {sys.version}")
        logger.info(f"TF version: {tf.__version__}")
        logger.info(f"Keras version: {tf.keras.__version__}")
        logger.info(f"OpenCV version: {cv.__version__}")
        logger.info(f"Numpy version: {np.__version__}")
        logger.info(f"Matplotlib version: {matplotlib.__version__}")
        logger.info(f"Loguru version: {loguru.__version__}")

        self.__version__ = "0.0.1"

        self._batch_size = 4
        self._img_height = 256
        self._img_width = 256
        self._epochs = 12

        self._AUTOTUNE = tf.data.AUTOTUNE

        self._class_names = []

        self._data_dir = None
        self._data_augmentation = True

        self._list_ds = None
        self._train_ds = None
        self._val_ds = None
        self._test_ds = None

        self._train_pourcent = 0.8
        self._val_pourcent = 0.1
        self._test_pourcent = 0.1

        self._history = None

        self._model = None
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        #self._loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        #self._metrics = tf.metrics.SparseCategoricalAccuracy()

        #self._loss = tf.losses.BinaryCrossentropy(from_logits=True)
        #self._metrics = tf.metrics.BinaryAccuracy()

        self._loss = tf.losses.CategoricalCrossentropy(from_logits=True)
        self._metrics = tf.metrics.CategoricalAccuracy()

        self._tf_callbacks = []

        for key, val in kwargs.items():
            self.__dict__[key] = val


if __name__ == '__main__':

    # Init AI
    ai = AI()

    parser = ArgumentParser()
    parser.add_argument("--display", action=argparse.BooleanOptionalAction,
                        default=True, help="Display the result")

    parser.add_argument("--data_augmentation", action=argparse.BooleanOptionalAction,
                        default=True, help="Use data augmentation")

    parser.add_argument("--gpu", action=argparse.BooleanOptionalAction,
                        default=True, help="Use GPU")

    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction,
                        default=False, help="Enable tensorboard")

    parser.add_argument("--checkpoint", action=argparse.BooleanOptionalAction,
                        default=False, help="Enable checkpoint")

    parser.add_argument("--continuous-integration", action=argparse.BooleanOptionalAction,
                        default=False, help="Enable continuous integration test")

    parser.add_argument("--load-model", type=str,
                        default=None, help="Load a model")
    parser.add_argument("--save-model", type=str,
                        default=None, help="Save a model")

    parser.add_argument("--epochs", type=int, default=ai._epochs,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int,
                        default=ai._batch_size, help="Batch size")
    parser.add_argument("--data_dir", type=str,
                        default=None, help="Data directory")
    parser.add_argument("--model_path", type=str,
                        default=None, help="Model path")
    #parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer")

    parser.add_argument("--class_names", type=str, nargs='+',
                        default=[], help="Class names")

    parser.add_argument("--train_pourcent", type=float,
                        default=ai._train_pourcent, help="Train pourcent")
    parser.add_argument("--val_pourcent", type=float,
                        default=ai._val_pourcent, help="Validation pourcent")
    parser.add_argument("--test_pourcent", type=float,
                        default=ai._test_pourcent, help="Test pourcent")

    parser.add_argument("--img_height", type=int,
                        default=ai._img_height, help="Image height")
    parser.add_argument("--img_width", type=int,
                        default=ai._img_width, help="Image width")

    parser.add_argument("--predict", type=str,
                        default=None, help="Predict image")

    #parser.add_argument("--loss", type=str, default="categorical_crossentropy", help="Loss")
    #parser.add_argument("--metrics", type=str, default="accuracy", help="Metrics")

    args = parser.parse_args()

    logger.debug(f"data_augmentation: {args.data_augmentation}")
    ai._data_augmentation = args.data_augmentation

    logger.debug(f"epochs: {args.epochs}")
    ai._epochs = args.epochs

    logger.debug(f"batch_size: {args.batch_size}")
    ai._batch_size = args.batch_size

    logger.debug(f"data_dir: {args.data_dir}")
    ai._data_dir = args.data_dir

    logger.debug(f"model_path: {args.model_path}")
    ai._model_path = args.model_path

    #logger.debug(f"optimizer: {args.optimizer}")
    #ai._optimizer = args.optimizer

    logger.debug(f"train_pourcent: {args.train_pourcent}")
    ai._train_pourcent = args.train_pourcent

    logger.debug(f"val_pourcent: {args.val_pourcent}")
    ai._val_pourcent = args.val_pourcent

    logger.debug(f"test_pourcent: {args.test_pourcent}")
    ai._test_pourcent = args.test_pourcent

    logger.debug(f"img_height: {args.img_height}")
    ai._img_height = args.img_height

    logger.debug(f"img_width: {args.img_width}")
    ai._img_width = args.img_width

    logger.debug(f"class_names: {args.class_names}")
    ai._class_names = args.class_names

    #logger.debug(f"loss: {args.loss}")
    #ai._loss = args.loss

    #logger.debug(f"metrics: {args.metrics}")
    #ai._metrics = args.metrics

    logger.debug(f"continuous_integration: {args.continuous_integration}")
    #ai._continuous_integration = args.continuous_integration

    if args.tensorboard:
        logger.debug("Enable tensorboard")
        log_dir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        logger.debug(f"Log dir: {log_dir}")

        # Create folder for logs if not exist
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ai._tf_callbacks.append(tf.keras.callbacks.TensorBoard(
            log_dir, histogram_freq=1, write_graph=True, write_images=True, update_freq=1))

    if args.checkpoint:
        logger.debug("Enable checkpoint")
        checkpoint_path = "./checkpoints/" + \
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        logger.debug(f"Checkpoint path: {checkpoint_path}")

        # Create folder for checkpoints if not exist
        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
        ai._tf_callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)
        )

    # Enable GPU
    if args.gpu:
        ai.gpu()
        logger.debug("Enable GPU support")

    if ai._data_dir is None:
        logger.warning("No data directory specified")
        logger.warning("Chose default dataset")
        data_dir = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
        logger.warning(f"Download dataset from {data_dir}")

        ai._data_dir = pathlib.Path(tf.keras.utils.get_file(
            'flower_photos', origin=data_dir, untar=True))

    if args.predict is None:
        ai.load_data()
        ai.prepare_train()

        if args.continuous_integration:
            ai._model = ai.get_ci_model()

    if args.load_model is not None:
        ai.load_model(args.load)
    else:
        ai.load_model()

    if args.predict is None:
        ai.compile()
        ai.train()
        ai.evaluate()

        if args.save_model is not None:
            ai.save_model(args.save)

        if args.display:
            ai.display_predict()
            ai.display_history()

    if args.predict is not None:
        ai.predict(args.predict)
