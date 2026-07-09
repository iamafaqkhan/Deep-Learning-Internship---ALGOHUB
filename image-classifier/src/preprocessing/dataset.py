"""
dataset.py
----------

Build TensorFlow datasets from image folders.

Responsibilities
----------------
- Load metadata from DatasetLoader
- Read images
- Apply preprocessing
- Apply augmentation (training only)
- Batch data
- Shuffle training data
- Prefetch data
"""

from pathlib import Path
import tensorflow as tf

from src.preprocessing.loader import DatasetLoader
from src.preprocessing.preprocess import ImagePreprocessor
from src.preprocessing.augmentation import ImageAugmentor


class DatasetBuilder:

    def __init__(
        self,
        dataset_path,
        image_size=(32, 32),
        batch_size=32,
        shuffle=True,
    ):

        self.batch_size = batch_size

        self.shuffle = shuffle

        self.loader = DatasetLoader(dataset_path)

        self.preprocessor = ImagePreprocessor(
            image_size=image_size
        )

        self.augmentor = ImageAugmentor()

    # --------------------------------------------------

    def _load_image(self, image_path, label):
        """
        Reads and preprocesses one image.
        """

        image = self.loader.read_image(image_path.numpy().decode())

        image = self.preprocessor.preprocess(image)

        return image, label

    # --------------------------------------------------

    def _tf_load_image(self, image_path, label):

        image, label = tf.py_function(
            self._load_image,
            [image_path, label],
            [tf.float32, tf.int32]
        )

        image.set_shape(
            (
                self.preprocessor.image_size[0],
                self.preprocessor.image_size[1],
                3
            )
        )

        label.set_shape(())

        return image, label

    # --------------------------------------------------

    def build(self, split="train"):

        image_paths, labels = self.loader.load_split(split)

        dataset = tf.data.Dataset.from_tensor_slices(
            (
                [str(p) for p in image_paths],
                labels
            )
        )

        dataset = dataset.map(
            self._tf_load_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        if split == "train":

            if self.shuffle:

                dataset = dataset.shuffle(
                    buffer_size=len(image_paths)
                )

            dataset = dataset.map(
                lambda x, y: (
                    self.augmentor.augment(
                        tf.expand_dims(x, 0),
                        training=True
                    )[0],
                    y
                ),
                num_parallel_calls=tf.data.AUTOTUNE
            )

        dataset = dataset.batch(
            self.batch_size
        )

        dataset = dataset.prefetch(
            tf.data.AUTOTUNE
        )

        return dataset

    # --------------------------------------------------

    def get_class_names(self):

        return self.loader.classes