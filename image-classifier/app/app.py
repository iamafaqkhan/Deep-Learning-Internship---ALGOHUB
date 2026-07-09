"""
Streamlit Web App for Image Classification
------------------------------------------

Features
--------
- Upload an image
- Display uploaded image
- Predict image class
- Show confidence score
- Show Top-3 predictions
- Display probability chart
"""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predict import Predictor

# ======================================================
# Page Configuration
# ======================================================

st.set_page_config(
    page_title="Image Classifier",
    page_icon="🖼️",
    layout="wide",
)

# ======================================================
# Constants
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "saved_models" / "best_model.keras"

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# ======================================================
# Load Predictor
# ======================================================

@st.cache_resource
def load_predictor():
    return Predictor(
        model_path=MODEL_PATH,
        class_names=CLASS_NAMES,
    )

predictor = load_predictor()

# ======================================================
# Sidebar
# ======================================================

st.sidebar.title("Image Classifier")

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
### About

This application classifies images into one of the
10 CIFAR-10 categories using a CNN built with TensorFlow/Keras.

**Model**

- Custom CNN
- TensorFlow/Keras
- Accuracy ≈ 77%

**Classes**

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck
"""
)

# ======================================================
# Main Page
# ======================================================

st.title("🖼️ Image Classification using CNN")

st.write(
    "Upload an image and let the trained CNN predict its class."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:

    temp_path = PROJECT_ROOT / "temp_image.png"

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    col1, col2 = st.columns([1, 1])

    with col1:

        st.subheader("Uploaded Image")

        st.image(
            temp_path,
            use_container_width=True,
        )

    with col2:

        if st.button("Predict"):

            with st.spinner("Predicting..."):

                result = predictor.predict(temp_path)

            st.success("Prediction Completed")

            st.metric(
                "Predicted Class",
                result["predicted_class"].title(),
            )

            st.metric(
                "Confidence",
                f"{result['confidence']:.2%}",
            )

            st.markdown("---")

            st.subheader("Top Predictions")

            df = pd.DataFrame(result["top_predictions"])

            df["confidence"] = (
                df["confidence"] * 100
            ).round(2)

            df.rename(
                columns={
                    "class": "Class",
                    "confidence": "Confidence (%)",
                },
                inplace=True,
            )

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Prediction Probabilities")

            probability_df = pd.DataFrame(
                {
                    "Class": CLASS_NAMES,
                    "Probability": result["probabilities"],
                }
            )

            probability_df = probability_df.sort_values(
                by="Probability",
                ascending=False,
            )

            st.bar_chart(
                probability_df.set_index("Class")
            )

    if temp_path.exists():
        temp_path.unlink()