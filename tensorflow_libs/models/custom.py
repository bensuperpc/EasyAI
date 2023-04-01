from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, MaxPool2D, Dropout, Rescaling


def ben_model_v1(input_shape=(224, 224, 3)):
    model = models.Sequential([
        Rescaling(1./255, input_shape=input_shape),
        Conv2D(48, kernel_size=(3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(96, kernel_size=(3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.1),
        Flatten(),
        Dense(384, activation='relu')
    ])
    return model


def ci_model_v1(input_shape=(224, 224, 3)):
    model = models.Sequential([
        Rescaling(1./255, input_shape=input_shape),
        Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.1),
        Flatten(),
        Dense(128, activation='relu')
    ])
    return model
