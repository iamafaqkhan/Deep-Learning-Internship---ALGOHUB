"""
model.py
---------
Defines the CNN architecture for handwritten digit recognition.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
)


def build_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Build and compile a CNN model.

    Parameters
    ----------
    input_shape : tuple
        Shape of input images.
    num_classes : int
        Number of output classes.

    Returns
    -------
    model : tensorflow.keras.Model
    """

    model = Sequential([

        # First Convolution Block
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
            input_shape=input_shape
        ),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),


        # Second Convolution Block
        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation="relu",
            padding="same"
        ),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),


        # Fully Connected Layers
        Flatten(),

        Dense(128, activation="relu"),

        Dropout(0.3),

        Dense(num_classes, activation="softmax")

    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model