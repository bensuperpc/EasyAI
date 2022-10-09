#!/usr/bin/env python

import argparse
from argparse import ArgumentParser

import os
import PIL
import pathlib
import datetime
import sys
from pathlib import Path
import time

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
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                logger.debug(
                    f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                logger.error(e)

    def get_label(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        one_hot = parts[-2] == self._class_names

        return tf.argmax(one_hot)

    def decode_img(self, img):
        img = tf.io.decode_jpeg(img, dct_method="INTEGER_ACCURATE", channels=3)

        return tf.image.resize(img, [self._img_height, self._img_width])

    def process_path(self, file_path):
        label = self.get_label(file_path)
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)

        return img, label

    def configure_for_performance(self, ds):
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(self._batch_size)

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

        ds = ds.prefetch(buffer_size=self._AUTOTUNE)
        return ds

    def get_model(self):
        # data_augmentation = tf.keras.Sequential([
        #  layers.RandomFlip("horizontal_and_vertical"),
        #  layers.RandomRotation(0.2),
        # ])

        model = Sequential([
            # data_augmentation,
            layers.Rescaling(1./255),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dense(len(self._class_names))
        ])

        model.build((None, self._img_height, self._img_width, 3))
        model.summary()

        return model

    def get_ci_model(self):
        model = Sequential([
            layers.Rescaling(1./255),
            layers.Conv2D(12, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(24, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(48, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(96, activation='relu'),
            layers.Dense(len(self._class_names))
        ])

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

            self._model.pop()
            self._model.add(layers.Dense(len(self._class_names)))
            #self._model.build((None, self._img_height, self._img_width, 3))
            # self._model.summary()

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
        acc = self._history.history['accuracy']
        val_acc = self._history.history['val_accuracy']

        loss = self._history.history['loss']
        val_loss = self._history.history['val_loss']

        epochs_range = range(self._epochs)

        plt.style.use('ggplot')

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

    def plot_image(self, predictions_array, true_label, img, grid=False, pred_color='red', true_color='blue'):
        plt.style.use('ggplot')
        plt.grid(grid)
        plt.xticks([])
        plt.yticks([])
        img = np.array(img/np.amax(img)*255, np.int32)
        plt.imshow(img, cmap=plt.cm.binary)
        predicted_label = np.argmax(predictions_array)

        if predicted_label == true_label:
            color = true_color
        else:
            color = pred_color
        plt.xlabel("{} {:2.0f}% ({})".format(self._class_names[predicted_label],
                                             100*np.max(predictions_array),
                                             self._class_names[true_label]),
                   color=color)

    def plot_value_array(self, predictions_array, true_label, grid=False, pred_color='red', true_color='blue'):
        plt.style.use('ggplot')
        plt.grid(grid)
        plt.xticks(range(len(self._class_names)))
        plt.yticks([])
        thisplot = plt.bar(range(len(self._class_names)),
                           predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)
        thisplot[predicted_label].set_color(pred_color)
        thisplot[true_label].set_color(true_color)

    def display_predict(self):
        image_batch, label_batch = next(iter(self._test_ds))

        probability_model = tf.keras.Sequential([self._model,
                                                tf.keras.layers.Softmax()])
        predictions = probability_model.predict(image_batch)

        num_rows = 5
        num_cols = 3
        num_images = num_rows*num_cols
        plt.figure(figsize=(2*2*num_cols, 2*num_rows))

        for i in range(num_images):
            _label_batch = label_batch[i]
            _label_batch = _label_batch.numpy().tolist()

            plt.subplot(num_rows, 2*num_cols, 2*i+1)
            self.plot_image(predictions[i], _label_batch, image_batch[i])

            plt.subplot(num_rows, 2*num_cols, 2*i+2)
            self.plot_value_array(predictions[i], _label_batch)

        plt.tight_layout()
        plt.show()

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

        self._batch_size = 24
        self._img_height = 256
        self._img_width = 256
        self._epochs = 8

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
        # tf.keras.optimizers.Adam(learning_rate=1e-4)
        self._optimizer = "adam"

        self._loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        self._metrics = ["accuracy"]

        self._tf_callbacks = []

        for key, val in kwargs.items():
            self.__dict__[key] = val

    @property
    def data_augmentation(self):
        return self._data_augmentation

    @data_augmentation.setter
    def data_augmentation(self, val):
        self._data_augmentation = val

    @data_augmentation.deleter
    def data_augmentation(self):
        del self._data_augmentation

    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def name(self, val):
        self._epochs = val

    @name.deleter
    def epochs(self):
        del self._epochs

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, val):
        self._batch_size = val

    @batch_size.deleter
    def batch_size(self):
        del self._batch_size

    @property
    def data_dir(self):
        return self._data_dir

    @data_dir.setter
    def name(self, val):
        self._data_dir = val

    @name.deleter
    def data_dir(self):
        del self._data_dir

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, val):
        self._optimizer = val

    @optimizer.deleter
    def optimizer(self):
        del self._optimizer

    @property
    def train_pourcent(self):
        return self._train_pourcent

    @train_pourcent.setter
    def train_pourcent(self, val):
        self._train_pourcent = val

    @train_pourcent.deleter
    def train_pourcent(self):
        del self._train_pourcent

    @property
    def val_pourcent(self):
        return self._val_pourcent

    @val_pourcent.setter
    def val_pourcent(self, val):
        self._val_pourcent = val

    @val_pourcent.deleter
    def val_pourcent(self):
        del self._val_pourcent

    @property
    def test_pourcent(self):
        return self._test_pourcent

    @test_pourcent.setter
    def test_pourcent(self, val):
        self._test_pourcent = val

    @test_pourcent.deleter
    def test_pourcent(self):
        del self._test_pourcent

    @property
    def img_height(self):
        return self._img_height

    @img_height.setter
    def img_height(self, val):
        self._img_height = val

    @img_height.deleter
    def img_height(self):
        del self._img_height

    @property
    def img_width(self):
        return self._img_width

    @img_width.setter
    def img_width(self, val):
        self._img_width = val

    @img_width.deleter
    def img_width(self):
        del self._img_width

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, val):
        self._loss = val

    @loss.deleter
    def loss(self):
        del self._loss

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, val):
        self._metrics = val

    @metrics.deleter
    def metrics(self):
        del self._metrics

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, val):
        self._model = val

    @model.deleter
    def model(self):
        del self._model

    @property
    def history(self):
        return self._history

    @history.setter
    def history(self, val):
        self._history = val

    @history.deleter
    def history(self):
        del self._history

    @property
    def tf_callbacks(self):
        return self._tf_callbacks

    @tf_callbacks.setter
    def tf_callbacks(self, val):
        self._tf_callbacks = val

    @tf_callbacks.deleter
    def tf_callbacks(self):
        del self._tf_callbacks

    @property
    def class_names(self):
        return self._class_names

    @class_names.setter
    def class_names(self, val):
        self._class_names = val

    @class_names.deleter
    def class_names(self):
        del self._class_names


if __name__ == '__main__':

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

    parser.add_argument("--load", type=str,
                        default=None, help="Load a model")
    parser.add_argument("--save", type=str,
                        default=None, help="Save a model")

    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int,
                        default=24, help="Batch size")
    parser.add_argument("--data_dir", type=str,
                        default=None, help="Data directory")
    parser.add_argument("--model_path", type=str,
                        default=None, help="Model path")
    #parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer")

    parser.add_argument("--class_names", type=str, nargs='+',
                        default=[], help="Class names")

    parser.add_argument("--train_pourcent", type=float,
                        default=0.8, help="Train pourcent")
    parser.add_argument("--val_pourcent", type=float,
                        default=0.1, help="Validation pourcent")
    parser.add_argument("--test_pourcent", type=float,
                        default=0.1, help="Test pourcent")

    parser.add_argument("--img_height", type=int,
                        default=256, help="Image height")
    parser.add_argument("--img_width", type=int,
                        default=256, help="Image width")

    parser.add_argument("--predict", type=str,
                        default=None, help="Predict image")

    #parser.add_argument("--loss", type=str, default="categorical_crossentropy", help="Loss")
    #parser.add_argument("--metrics", type=str, default="accuracy", help="Metrics")

    args = parser.parse_args()

    # Init AI
    ai = AI()

    logger.debug(f"data_augmentation: {args.data_augmentation}")
    ai.data_augmentation = args.data_augmentation

    logger.debug(f"epochs: {args.epochs}")
    ai.epochs = args.epochs

    logger.debug(f"batch_size: {args.batch_size}")
    ai.batch_size = args.batch_size

    logger.debug(f"data_dir: {args.data_dir}")
    ai.data_dir = args.data_dir

    logger.debug(f"model_path: {args.model_path}")
    ai.model_path = args.model_path

    #logger.debug(f"optimizer: {args.optimizer}")
    #ai.optimizer = args.optimizer

    logger.debug(f"train_pourcent: {args.train_pourcent}")
    ai.train_pourcent = args.train_pourcent

    logger.debug(f"val_pourcent: {args.val_pourcent}")
    ai.val_pourcent = args.val_pourcent

    logger.debug(f"test_pourcent: {args.test_pourcent}")
    ai.test_pourcent = args.test_pourcent

    logger.debug(f"img_height: {args.img_height}")
    ai.img_height = args.img_height

    logger.debug(f"img_width: {args.img_width}")
    ai.img_width = args.img_width

    logger.debug(f"class_names: {args.class_names}")
    ai.class_names = args.class_names

    #logger.debug(f"loss: {args.loss}")
    #ai.loss = args.loss

    #logger.debug(f"metrics: {args.metrics}")
    #ai.metrics = args.metrics

    logger.debug(f"continuous_integration: {args.continuous_integration}")
    #ai.continuous_integration = args.continuous_integration

    if args.tensorboard:
        logger.debug("Enable tensorboard")
        log_dir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        logger.debug(f"Log dir: {log_dir}")

        # Create folder for logs if not exist
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ai.tf_callbacks.append(tf.keras.callbacks.TensorBoard(
            log_dir, histogram_freq=1, write_graph=True, write_images=True, update_freq=1))

    if args.checkpoint:
        logger.debug("Enable checkpoint")
        checkpoint_path = "./checkpoints/" + \
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        logger.debug(f"Checkpoint path: {checkpoint_path}")

        # Create folder for checkpoints if not exist
        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
        ai.tf_callbacks.append(tf.keras.callbacks.ModelCheckpoint(
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

    if ai.data_dir is None:
        logger.warning("No data directory specified")
        logger.warning("Chose default dataset")
        data_dir = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
        logger.warning(f"Download dataset from {data_dir}")

        ai.data_dir = pathlib.Path(tf.keras.utils.get_file(
            'flower_photos', origin=data_dir, untar=True))

    if args.predict is None:
        ai.load_data()
        ai.prepare_train()

        if args.continuous_integration:
            ai.model = ai.get_ci_model()

    if args.load is not None:
        ai.load_model(args.load)
    else:
        ai.load_model()

    if args.predict is None:
        ai.compile()
        ai.train()
        ai.evaluate()

        if args.save is not None:
            ai.save_model(args.save)

        if args.display:
            ai.display_predict()
            ai.display_history()

    if args.predict is not None:

        ai.predict(args.predict)
