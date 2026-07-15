import tensorflow as tf
from keras.models import Sequential
from keras.layers import (
    Input,
    Rescaling,
    Conv2D,
    MaxPooling2D,
    Dense,
    Dropout,
    BatchNormalization,
    RandomFlip,
    RandomRotation,
    RandomZoom,
    GlobalAveragePooling2D
)

IMAGE_SIZE = (128, 128)

data_augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1)

])

def build_model():

    model = Sequential([

        Input(shape=(128, 128, 3)),

        data_augmentation,
        
        Rescaling(1./255),
        
        Conv2D(
            16,
            (3,3),
            activation="relu",
            padding="same"
        ),

        BatchNormalization(),
        MaxPooling2D(),

        Conv2D(
            32,
            (3,3),
            activation="relu",
            padding="same"
        ),

        BatchNormalization(),
        MaxPooling2D(),

        Conv2D(
            64,
            (3,3),
            activation="relu",
            padding="same"
        ),

        BatchNormalization(),
        MaxPooling2D(),

        Conv2D(
            128,
            (3,3),
            activation="relu",
            padding="same"
        ),

        BatchNormalization(),
        MaxPooling2D(),

        GlobalAveragePooling2D(),

        Dense(128, activation="relu"),
        Dropout(0.4),

        Dense(6, activation="softmax")

    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model