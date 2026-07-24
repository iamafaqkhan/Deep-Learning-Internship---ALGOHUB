"""
Predict Script
--------------
Predict the disease class for a single plant leaf image.

Author : Afaq Ahmad Khan
Project: Plant Disease Prediction
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from src.config import (
    TRAIN_DIR,
    MODEL_PATH,
    IMAGE_SIZE,
)

# ==============================================================================
# Load Model
# ==============================================================================

if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

# ==============================================================================
# Class Names
# ==============================================================================

CLASS_NAMES = sorted(
    [
        folder.name
        for folder in Path(TRAIN_DIR).iterdir()
        if folder.is_dir()
    ]
)

# ==============================================================================
# Predict Function
# ==============================================================================

def predict_image(image_path):
    """
    Predict disease from a plant leaf image.

    Args:
        image_path (str): Path to the input image.

    Returns:
        tuple:
            predicted_class (str)
            confidence (float)
    """

    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image = np.array(image, dtype=np.float32)
    image = np.expand_dims(image, axis=0)

    image = tf.keras.applications.efficientnet.preprocess_input(image)

    prediction = model.predict(image, verbose=0)

    predicted_index = np.argmax(prediction)

    confidence = float(np.max(prediction))

    predicted_class = CLASS_NAMES[predicted_index]

    return predicted_class, confidence


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":

    image_path = input("Enter image path: ").strip()

    predicted_class, confidence = predict_image(image_path)

    print("\nPrediction")
    print("-" * 40)
    print(f"Disease   : {predicted_class}")
    print(f"Confidence: {confidence:.2%}")