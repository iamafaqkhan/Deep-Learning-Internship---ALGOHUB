"""
model.py
--------

Convolutional Neural Network (CNN) for image classification.

Responsibilities
----------------
- Build CNN architecture
- Compile the model
- Display model summary

Author: Afaq Ahmad Khan
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Flatten,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
)
from tensorflow.keras.optimizers import Adam


class ImageClassifier:

    def __init__(
        self,
        input_shape=(32, 32, 3),
        num_classes=10,
        learning_rate=1e-3,
    ):

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        self.model = self.build_model()
        self.compile_model()

    # -----------------------------------------------------

    def build_model(self) -> Model:
        """
        Build CNN architecture.
        """

        inputs = Input(shape=self.input_shape)

        # Block 1
        x = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
        )(inputs)

        x = BatchNormalization()(x)

        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Block 2
        x = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
        )(x)

        x = BatchNormalization()(x)

        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Block 3
        x = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
        )(x)

        x = BatchNormalization()(x)

        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Add Block 4 to overcome underfitting
        x = Conv2D(
            filters=256,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
        )(x)

        x = BatchNormalization()(x)

        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Classification Head

        #x = Flatten()(x)
        x = GlobalAveragePooling2D()(x)

        x = Dense(
            units=256,
            activation="relu",
        )(x)

        x = Dropout(0.5)(x)

        outputs = Dense(
            units=self.num_classes,
            activation="softmax",
        )(x)

        model = Model(
            inputs=inputs,
            outputs=outputs,
            name="Image_Classifier_CNN",
        )

        return model

    # -----------------------------------------------------

    def compile_model(self):
        """
        Compile the CNN.
        """

        self.model.compile(
            optimizer=Adam(
                learning_rate=self.learning_rate
            ),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

    # -----------------------------------------------------

    def summary(self):
        """
        Display model architecture.
        """

        return self.model.summary()

    # -----------------------------------------------------

    def get_model(self):
        """
        Return compiled model.
        """

        return self.model