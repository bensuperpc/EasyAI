import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

def display_history(_history, _epochs):
    if 'accuracy' in _history.history:
        acc = _history.history['accuracy']
        val_acc = _history.history['val_accuracy']
    elif 'binary_accuracy' in _history.history:
        acc = _history.history['binary_accuracy']
        val_acc = _history.history['val_binary_accuracy']
    elif 'sparse_categorical_accuracy' in _history.history:
        acc = _history.history['sparse_categorical_accuracy']
        val_acc = _history.history['val_sparse_categorical_accuracy']
    elif 'categorical_accuracy' in _history.history:
        acc = _history.history['categorical_accuracy']
        val_acc = _history.history['val_categorical_accuracy']
    else:
        logger.error("No compatible accuracy key found in history")
        return

    loss = _history.history['loss']
    val_loss = _history.history['val_loss']

    epochs_range = range(_epochs)

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

def plot_image(predictions_array, true_label, img, grid=False, pred_color='red', true_color='blue', _class_names=None):
    plt.style.use('ggplot')
    plt.grid(grid)
    plt.xticks([])
    plt.yticks([])
    img = np.array(img/np.amax(img)*255, np.int32)
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if list(true_label)[predicted_label]:
        color = true_color
    else:
        color = pred_color
    #plt.xlabel("{} {:2.0f}% ({})".format(_class_names[predicted_label],
    #                                     100*np.max(predictions_array),
    #                                     _class_names[true_label]),
    #           color=color)
    plt.xlabel("{}".format(_class_names[true_label]),
                color=color)

def plot_value_array(predictions_array, true_label, grid=False, pred_color='red', true_color='blue', _class_names=None):
    plt.style.use('ggplot')
    plt.grid(grid)
    plt.xticks(range(len(_class_names)))
    plt.yticks([])
    thisplot = plt.bar(range(len(_class_names)),
                        predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color(pred_color)

    for i in range(len(_class_names)):
        if list(true_label)[i]:
            thisplot[i].set_color(true_color)

def display_predict(_model, _dataset, _class_names, num_rows=5, num_cols=3):

    num_images = num_rows*num_cols

    image_batch, label_batch = next(iter(_dataset))

    while len(label_batch) < num_images:
        new_image_batch, new_label_batch = next(iter(_dataset))
        image_batch = tf.concat([image_batch, new_image_batch], axis=0)
        label_batch = tf.concat([label_batch, new_label_batch], axis=0)

    probability_model = tf.keras.Sequential([_model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(image_batch)

    plt.figure(figsize=(2*2*num_cols, 2*num_rows))

    for i in range(num_images):
        _label_batch = label_batch[i]
        _label_batch = _label_batch.numpy().tolist()

        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(predictions[i], _label_batch, image_batch[i], _class_names=_class_names)

        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(predictions[i], _label_batch, _class_names=_class_names)

    plt.tight_layout()
    plt.show()