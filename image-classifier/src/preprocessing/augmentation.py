"""
augmentation.py
----------------

Image augmentation pipeline using TensorFlow/Keras.

Responsibilities
----------------
- Random Flip
- Random Rotation
- Random Zoom
- Random Translation
- Random Contrast

Applied ONLY during training.
"""

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    RandomFlip,
    RandomRotation,
    RandomZoom,
    RandomTranslation,
    RandomContrast,
)


class ImageAugmentor:
    """
    Image augmentation pipeline.
    """

    def __init__(
        self,
        flip="horizontal",
        rotation=0.1,
        zoom=0.1,
        translation=0.1,
        contrast=0.1,
    ):

        self.pipeline = Sequential(
            [
                RandomFlip(mode=flip),

                RandomRotation(
                    factor=rotation
                ),

                RandomZoom(
                    height_factor=zoom,
                    width_factor=zoom
                ),

                RandomTranslation(
                    height_factor=translation,
                    width_factor=translation
                ),

                RandomContrast(
                    factor=contrast
                ),
            ],
            name="augmentation_pipeline"
        )

    # -----------------------------------------------------

    def augment(self, images, training=True):
        """
        Apply augmentation.

        Parameters
        ----------
        images : Tensor or NumPy array

        training : bool
            True during training
            False during validation/testing

        Returns
        -------
        Augmented images
        """

        return self.pipeline(
            images,
            training=training
        )

    # -----------------------------------------------------

    def summary(self):
        """
        Print augmentation pipeline.
        """

        self.pipeline.summary()