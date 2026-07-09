"""
predict.py
----------

Inference module for the Image Classifier.

Responsibilities
----------------
- Load trained model
- Preprocess input image
- Predict image class
- Return Top-K predictions
"""

from pathlib import Path
import sys

import cv2
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.preprocess import ImagePreprocessor


class Predictor:
    """
    CNN inference class.
    """

    def __init__(
        self,
        model_path,
        class_names,
        image_size=(32, 32),
    ):
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = class_names
        self.preprocessor = ImagePreprocessor(image_size=image_size)

    # ---------------------------------------------------------

    def _load_image(self, image_path):
        """
        Read and preprocess an image.
        """

        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        processed = self.preprocessor.preprocess(image)

        return image, processed

    # ---------------------------------------------------------

    def predict(self, image_path, top_k=3):
        """
        Predict the class of an image.

        Parameters
        ----------
        image_path : str or Path
            Path to the input image.

        top_k : int
            Number of top predictions to return.

        Returns
        -------
        dict
            Prediction results.
        """

        original_image, processed_image = self._load_image(image_path)

        processed_image = np.expand_dims(
            processed_image,
            axis=0,
        )

        probabilities = self.model.predict(
            processed_image,
            verbose=0,
        )[0]

        predicted_index = np.argmax(probabilities)

        predicted_class = self.class_names[predicted_index]

        confidence = float(probabilities[predicted_index])

        top_indices = np.argsort(probabilities)[::-1][:top_k]

        top_predictions = []

        for index in top_indices:
            top_predictions.append(
                {
                    "class": self.class_names[index],
                    "confidence": float(probabilities[index]),
                }
            )

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "top_predictions": top_predictions,
            "probabilities": probabilities,
            "image": original_image,
        }