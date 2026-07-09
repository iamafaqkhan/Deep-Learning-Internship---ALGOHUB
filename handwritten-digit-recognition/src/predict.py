"""
predict.py
-----------
Functions for loading the trained model and making predictions.
"""

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps


# Load trained model
# def load_trained_model(model_path="model/mnist_cnn.keras"):
#     """
#     Load the trained CNN model.
#     """

#     model = tf.keras.models.load_model(model_path)
#     return model

from pathlib import Path
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "mnist_cnn.keras"


def load_trained_model():
    print(f"Looking for model at: {MODEL_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}"
        )

    return tf.keras.models.load_model(MODEL_PATH)


# Preprocess uploaded image
def preprocess_uploaded_image(image):
    """
    Convert uploaded image into the same format used during training.

    Steps:
    - Convert to grayscale
    - Resize to 28x28
    - Invert colors
    - Normalize
    - Reshape
    """

    # Convert to grayscale
    image = image.convert("L")

    # Resize
    image = image.resize((28, 28))

    # Invert image
    image = ImageOps.invert(image)

    # Convert to numpy
    image = np.array(image)

    # Normalize
    image = image.astype("float32") / 255.0

    # Add channel dimension
    image = np.expand_dims(image, axis=-1)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image


# Predict digit
def predict_digit(model, image):
    """
    Predict handwritten digit.
    """

    predictions = model.predict(image, verbose=0)

    predicted_digit = np.argmax(predictions)

    confidence = predictions[0][predicted_digit]

    return predicted_digit, confidence, predictions[0]