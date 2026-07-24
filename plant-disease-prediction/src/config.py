"""
Project Configuration
---------------------
Centralized configuration for the Plant Disease Prediction project.

Author : Afaq Ahmad Khan
Project: Plant Disease Prediction
"""

from pathlib import Path

# ==============================================================================
# Project Paths
# ==============================================================================

# Project Root
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VALID_DIR = DATA_DIR / "valid"
TEST_DIR = DATA_DIR / "test"

# Models
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "best_model.keras"

# ==============================================================================
# Dataset Parameters
# ==============================================================================

# Image size expected by EfficientNetB0
IMAGE_SIZE = (224, 224)

# RGB images
CHANNELS = 3

# Number of classes
# Leave None. It will be determined automatically from the dataset.
NUM_CLASSES = None

# Batch size
BATCH_SIZE = 32

# Random seed
SEED = 42

# ==============================================================================
# Training Parameters
# ==============================================================================

# Phase 1 (Feature Extraction)
INITIAL_EPOCHS = 10

# Phase 2 (Fine-Tuning)
FINE_TUNE_EPOCHS = 1

# Total epochs
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS

# Learning Rates
INITIAL_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5

# ==============================================================================
# Callbacks
# ==============================================================================

# EarlyStopping
PATIENCE = 5

# ReduceLROnPlateau
LR_FACTOR = 0.2
LR_PATIENCE = 2

# ==============================================================================
# Transfer Learning
# ==============================================================================

MODEL_NAME = "EfficientNetB0"

# Number of layers to keep trainable during fine-tuning
FINE_TUNE_AT = 11

# ==============================================================================
# Performance
# ==============================================================================

AUTOTUNE = "AUTOTUNE"

# ==============================================================================
# Supported Image Extensions
# ==============================================================================

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
)

# ==============================================================================
# Prediction
# ==============================================================================

CONFIDENCE_THRESHOLD = 0.50