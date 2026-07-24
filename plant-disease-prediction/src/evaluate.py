"""
Evaluate Script
---------------
Evaluate the trained model on the test dataset.

Author : Afaq Ahmad Khan
Project: Plant Disease Prediction
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

from src.config import (
    TEST_DIR,
    MODEL_PATH,
    IMAGE_SIZE,
    BATCH_SIZE,
)


# Load Test Dataset


test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False,
)

class_names = test_dataset.class_names

test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)


# Load Model


if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)


# Evaluate Model

test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1,
)

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")


# Predictions


y_true = []
y_pred = []

for images, labels in test_dataset:

    predictions = model.predict(images, verbose=0)

    y_pred.extend(np.argmax(predictions, axis=1))
    y_true.extend(np.argmax(labels.numpy(), axis=1))

# Classification Report


print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
    )
)