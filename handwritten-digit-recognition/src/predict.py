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
    Preprocess uploaded or drawn image so it closely matches
    the MNIST dataset format.

    Steps:
    1. Convert to grayscale
    2. Invert colors
    3. Crop to the digit's bounding box
    4. Resize while keeping aspect ratio
    5. Center on a 28x28 canvas
    6. Normalize
    7. Add batch and channel dimensions
    """

    

    # Convert to grayscale
    image = image.convert("L")

    # Invert colors (MNIST = white digit on black background)
    image = ImageOps.invert(image)

    image_np = np.array(image)

    # Threshold to remove background noise
    image_np = np.where(image_np > 30, image_np, 0)

    # Find the digit bounding box
    coords = np.argwhere(image_np > 0)

    if len(coords) == 0:
        # Blank image
        image_np = np.zeros((28, 28), dtype=np.float32)

    else:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1

        image_np = image_np[y0:y1, x0:x1]

        image = Image.fromarray(image_np)

        # Keep aspect ratio
        image.thumbnail((20, 20))

        # Create a black 28x28 canvas
        canvas = Image.new("L", (28, 28), color=0)

        x = (28 - image.width) // 2
        y = (28 - image.height) // 2

        canvas.paste(image, (x, y))

        image_np = np.array(canvas).astype("float32")

    # Normalize
    image_np /= 255.0

    # Add channel dimension
    image_np = np.expand_dims(image_np, axis=-1)

    # Add batch dimension
    image_np = np.expand_dims(image_np, axis=0)

    return image_np


# Predict digit
def predict_digit(model, image):
    """
    Predict handwritten digit.
    """

    predictions = model.predict(image, verbose=0)

    predicted_digit = np.argmax(predictions)

    confidence = predictions[0][predicted_digit]

    return predicted_digit, confidence, predictions[0]