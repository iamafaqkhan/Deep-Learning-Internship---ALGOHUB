"""
preprocess.py
-------------

Image preprocessing utilities.

Responsibilities:
- Resize images
- Normalize pixel values
- Convert image datatype
- Encode labels

Author: Afaq Ahmad Khan
"""

from pathlib import Path
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical


class ImagePreprocessor:
    """
    Image preprocessing pipeline.
    """

    def __init__(
        self,
        image_size=(32, 32),
        normalize=True
    ):
        self.image_size = image_size
        self.normalize = normalize

    # ---------------------------------------------------------

    def resize(self, image):
        """
        Resize image to target size.
        """

        return cv2.resize(
            image,
            self.image_size,
            interpolation=cv2.INTER_AREA
        )

    # ---------------------------------------------------------

    def normalize_image(self, image):
        """
        Normalize image from [0,255] to [0,1].
        """

        image = image.astype(np.float32)

        image /= 255.0

        return image

    # ---------------------------------------------------------

    def preprocess(self, image):
        """
        Complete preprocessing pipeline.
        """

        image = self.resize(image)

        if self.normalize:
            image = self.normalize_image(image)

        return image

    # ---------------------------------------------------------

    def preprocess_batch(self, images):
        """
        Preprocess a batch of images.
        """

        processed = [
            self.preprocess(img)
            for img in images
        ]

        return np.array(processed)

    # ---------------------------------------------------------

    @staticmethod
    def encode_labels(labels, num_classes):
        """
        Convert integer labels to one-hot vectors.
        """

        return to_categorical(
            labels,
            num_classes=num_classes
        )

    # ---------------------------------------------------------

    @staticmethod
    def decode_label(one_hot_label):
        """
        Convert one-hot vector back to class index.
        """

        return np.argmax(one_hot_label)