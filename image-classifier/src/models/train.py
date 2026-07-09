"""
train.py
--------

Train the CNN image classifier.

Responsibilities
----------------
- Load datasets
- Build CNN model
- Configure callbacks
- Train the model
- Save best model
- Save training history

Author: Afaq Ahmad Khan
"""

from pathlib import Path
import sys
import json

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    CSVLogger,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.dataset import DatasetBuilder
from src.models.model import ImageClassifier


# Configuration

DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "cifar10"

MODEL_DIR = PROJECT_ROOT / "saved_models"

REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(exist_ok=True)

REPORT_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = (32, 32)

BATCH_SIZE = 32

EPOCHS = 50

NUM_CLASSES = 10

LEARNING_RATE = 1e-3


# Dataset


print("Loading datasets...")

builder = DatasetBuilder(
    dataset_path=DATASET_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

train_dataset = builder.build(split="train")

test_dataset = builder.build(split="test")



# Model


print("Building model...")

classifier = ImageClassifier(
    input_shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3,
    ),
    num_classes=NUM_CLASSES,
    learning_rate=LEARNING_RATE,
)

model = classifier.get_model()

classifier.summary()



# Callbacks


callbacks = [

    EarlyStopping(

        monitor="val_loss",

        patience=5,

        restore_best_weights=True,

        verbose=1,
    ),

    ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.2,

        patience=3,

        verbose=1,
    ),

    ModelCheckpoint(

        filepath=MODEL_DIR / "best_model.keras",

        monitor="val_accuracy",

        save_best_only=True,

        verbose=1,
    ),

    CSVLogger(

        REPORT_DIR / "training_log.csv"
    ),
]



# Training


print("Training started...")

history = model.fit(

    train_dataset,

    validation_data=test_dataset,

    epochs=EPOCHS,

    callbacks=callbacks,

)



# Save Final Model


model.save(
    MODEL_DIR / "final_model.keras"
)



# Save History


history_path = REPORT_DIR / "history.json"

history_dict = history.history

with open(history_path, "w") as file:

    json.dump(
        history_dict,
        file,
        indent=4,
    )


print()

print("=" * 60)

print("Training Complete")

print(f"Best Model : {MODEL_DIR/'best_model.keras'}")

print(f"Final Model: {MODEL_DIR/'final_model.keras'}")

print(f"History    : {history_path}")

print("=" * 60)