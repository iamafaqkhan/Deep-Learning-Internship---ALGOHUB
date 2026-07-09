"""
evaluate.py
-----------

Evaluate the trained CNN model.

Responsibilities
----------------
- Load trained model
- Evaluate on test dataset
- Generate confusion matrix
- Generate classification report
- Plot training curves
- Save evaluation results

Author: Afaq Ahmad Khan
"""

from pathlib import Path
import sys
import json

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.dataset import DatasetBuilder


# Paths

DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "cifar10"

MODEL_PATH = PROJECT_ROOT / "saved_models" / "best_model.keras"

REPORT_PATH = PROJECT_ROOT / "reports"

HISTORY_PATH = REPORT_PATH / "history.json"


# Dataset


builder = DatasetBuilder(
    dataset_path=DATASET_PATH,
    batch_size=32,
)

test_dataset = builder.build("test")

class_names = builder.get_class_names()



# Load Model


model = tf.keras.models.load_model(MODEL_PATH)



# Evaluate


loss, accuracy = model.evaluate(test_dataset, verbose=1)

print(f"\nTest Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy:.4f}")



# Predictions


y_true = []
y_pred = []

for images, labels in test_dataset:

    predictions = model.predict(images, verbose=0)

    predictions = np.argmax(predictions, axis=1)

    y_pred.extend(predictions)

    y_true.extend(labels.numpy())



# Classification Report


report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
)

print("\nClassification Report\n")
print(report)


# Save report

with open(REPORT_PATH / "classification_report.txt", "w") as f:

    f.write(report)



# Confusion Matrix


cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)

fig, ax = plt.subplots(figsize=(10, 10))

disp.plot(
    cmap="Blues",
    xticks_rotation=45,
    ax=ax,
)

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(REPORT_PATH / "confusion_matrix.png")

plt.show()



# Training Curves


if HISTORY_PATH.exists():

    with open(HISTORY_PATH) as f:

        history = json.load(f)

    # Accuracy

    plt.figure(figsize=(8, 5))

    plt.plot(history["accuracy"], label="Train")

    plt.plot(history["val_accuracy"], label="Validation")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.title("Training Accuracy")

    plt.legend()

    plt.grid(True)

    plt.savefig(REPORT_PATH / "accuracy_curve.png")

    plt.show()

    # Loss

    plt.figure(figsize=(8, 5))

    plt.plot(history["loss"], label="Train")

    plt.plot(history["val_loss"], label="Validation")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Training Loss")

    plt.legend()

    plt.grid(True)

    plt.savefig(REPORT_PATH / "loss_curve.png")

    plt.show()

print("\nEvaluation completed successfully.")