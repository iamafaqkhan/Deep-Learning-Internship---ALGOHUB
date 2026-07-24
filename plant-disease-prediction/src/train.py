"""
Train Script
------------
Phase 2: Dataset Loading and Model Creation

Author : Afaq Ahmad Khan
Project: Plant Disease Prediction
"""

import tensorflow as tf
from pathlib import Path


from src.config import (
    TRAIN_DIR,
    VALID_DIR,
    TEST_DIR,
    MODEL_PATH,
    IMAGE_SIZE,
    BATCH_SIZE,
    SEED,
    INITIAL_EPOCHS,
    FINE_TUNE_EPOCHS,
    TOTAL_EPOCHS,
    INITIAL_LEARNING_RATE,
    FINE_TUNE_LEARNING_RATE,
    PATIENCE,
    LR_FACTOR,
    LR_PATIENCE,
    FINE_TUNE_AT,
)


# ==============================================================================
# Load Dataset
# ==============================================================================

def load_datasets():

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True,
        seed=SEED,
    )

    valid_dataset = tf.keras.utils.image_dataset_from_directory(
        VALID_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
    )

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
    )

    # Save class names BEFORE any dataset transformations
    class_names = train_dataset.class_names

    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_dataset.prefetch(AUTOTUNE)
    valid_dataset = valid_dataset.prefetch(AUTOTUNE)
    test_dataset = test_dataset.prefetch(AUTOTUNE)

    return train_dataset, valid_dataset, test_dataset, class_names

# ==============================================================================
# Build Model
# ==============================================================================

def build_model(num_classes):

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
        ],
        name="data_augmentation",
    )

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
        name="efficientnetb0",
    )

    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))

    x = data_augmentation(inputs)

    x = tf.keras.applications.efficientnet.preprocess_input(x)

    x = base_model(x, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
    )(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=INITIAL_LEARNING_RATE
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model

# ==============================================================================
# Callbacks
# ==============================================================================



def get_callbacks():

    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        verbose=1,
    )

    return [
        checkpoint,
        early_stopping,
        reduce_lr,
    ]

# ==============================================================================
# Feature Extraction
# ==============================================================================

def train_feature_extractor(model, train_ds, valid_ds):

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=INITIAL_EPOCHS,
        callbacks=get_callbacks(),
    )

    return history

# ==============================================================================
# Fine-Tuning
# ==============================================================================

def fine_tune_model(model, train_ds, valid_ds):

    base_model = model.get_layer("efficientnetb0")

    base_model.trainable = True

    for layer in base_model.layers[:-FINE_TUNE_AT]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=FINE_TUNE_LEARNING_RATE
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=TOTAL_EPOCHS,
        initial_epoch=INITIAL_EPOCHS,
        callbacks=get_callbacks(),
    )

    return history
# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":

    train_ds, valid_ds, test_ds, class_names = load_datasets()

    model = build_model(len(class_names))

    print(model.summary())

    print("\nStarting Feature Extraction...\n")

    feature_history = train_feature_extractor(
        model,
        train_ds,
        valid_ds,
    )

    fine_tune_history = fine_tune_model(
        model,
        train_ds,
        valid_ds,
    )

    print("\nTraining Completed Successfully!")