from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, MaxPool2D, Dropout, Rescaling
import tensorflow as tf
from loguru import logger


class ben_model_v1(tf.keras.Sequential):
    def __init__(self, **kwargs):

        layers = kwargs.get("layers", None)

        if layers is not None:
            logger.warning(
                "Layers will be ignored, use add()/pop() method instead")
        name = kwargs.get("name", None)

        super().__init__(layers=None, name=name)

        input_shape = kwargs.get("input_shape", (256, 256, 3))
        logger.debug(f"input_shape: {input_shape}")
        ouput_shape = kwargs.get("output_shape", 2)
        logger.debug(f"output_shape: {ouput_shape}")
        multiplier = kwargs.get("multiplier", 1)
        logger.debug(f"multiplier: {multiplier}")

        self.add(Rescaling(1./255, input_shape=input_shape))
        self.add(Conv2D(64 * multiplier, kernel_size=(3, 3),
                 padding='same', activation='relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Conv2D(128 * multiplier, kernel_size=(3, 3),
                 padding='same', activation='relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Conv2D(192 * multiplier, kernel_size=(3, 3),
                 padding='same', activation='relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Dropout(0.1))
        self.add(Flatten())
        self.add(Dense(384 * multiplier, activation='relu'))

        if ouput_shape >= 2:
            self.add(Dense(ouput_shape, activation='softmax'))
        else:
            self.add(Dense(ouput_shape, activation='sigmoid'))

    def build(self, input_shape=None):
        return super().build(input_shape)

    def call(self, inputs, training=None, mask=None):
        return super().call(inputs, training, mask)

    def add(self, layer):
        return super().add(layer)

    def pop(self):
        return super().pop()

    def seve_model(self, filepath, overwrite=True, include_optimizer=True, save_format=None, signatures=None, options=None, save_traces=True):
        return super().save(filepath, overwrite, include_optimizer, save_format, signatures, options)

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        return super().save_weights(filepath, overwrite, save_format, options)

    def load_weights(self, filepath, by_name=False, skip_mismatch=False, options=None):
        return super().load_weights(filepath, by_name, skip_mismatch, options)

    def compile(self, optimizer, loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, jit_compile=None,**kwargs):
        return super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, jit_compile=None, **kwargs)

    def evaluate(self, x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=True, return_dict=False, **kwargs):
        return super().evaluate(x, y, batch_size, verbose, sample_weight, steps, callbacks, max_queue_size, workers, use_multiprocessing, return_dict, **kwargs)
    

class ci_model_v1(tf.keras.Sequential):
    def __init__(self, **kwargs):

        layers = kwargs.get("layers", None)

        if layers is not None:
            logger.warning(
                "Layers will be ignored, use add()/pop() method instead")
        name = kwargs.get("name", None)

        super().__init__(layers=None, name=name)

        input_shape = kwargs.get("input_shape", (256, 256, 3))
        logger.debug(f"input_shape: {input_shape}")
        ouput_shape = kwargs.get("output_shape", 2)
        logger.debug(f"output_shape: {ouput_shape}")
        multiplier = kwargs.get("multiplier", 1)
        logger.debug(f"multiplier: {multiplier}")

        self.add(Rescaling(1./255, input_shape=input_shape))
        self.add(Conv2D(16 * multiplier, kernel_size=(3, 3),
                 padding='same', activation='relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Conv2D(32 * multiplier, kernel_size=(3, 3),
                 padding='same', activation='relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Conv2D(64 * multiplier, kernel_size=(3, 3),
                 padding='same', activation='relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Dropout(0.1))
        self.add(Flatten())
        self.add(Dense(128 * multiplier, activation='relu'))

        if ouput_shape >= 2:
            self.add(Dense(ouput_shape, activation='softmax'))
        else:
            self.add(Dense(ouput_shape, activation='sigmoid'))
