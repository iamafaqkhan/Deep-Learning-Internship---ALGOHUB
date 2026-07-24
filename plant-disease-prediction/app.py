"""
Streamlit Application
---------------------
Plant Disease Prediction using EfficientNetB0.

Author : Afaq Ahmad Khan
Project: Plant Disease Prediction
"""

import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

from src.predict import predict_image


# ==============================================================================
# Page Configuration
# ==============================================================================

st.set_page_config(
    page_title="Plant Disease Prediction",
    page_icon="🌿",
    layout="centered",
)

# ==============================================================================
# Title
# ==============================================================================

st.title("Plant Disease Prediction")

st.write(
    """
Upload a clear image of a plant leaf to identify its disease using a
deep learning model based on EfficientNetB0.
"""
)

st.divider()

# ==============================================================================
# File Upload
# ==============================================================================

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"],
)

# ==============================================================================
# Prediction
# ==============================================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True,
    )

    if st.button("Predict"):

        with st.spinner("Analyzing image..."):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=Path(uploaded_file.name).suffix,
            ) as temp_file:

                image.save(temp_file.name)

                predicted_class, confidence = predict_image(temp_file.name)

        st.divider()

        st.subheader("Prediction Result")

        st.write(f"**Disease:** {predicted_class}")

        st.write(f"**Confidence:** {confidence:.2%}")

        if confidence >= 0.90:
            st.success("Prediction confidence is high.")
        elif confidence >= 0.70:
            st.info("Prediction confidence is moderate.")
        else:
            st.warning(
                "Prediction confidence is low. Consider using a clearer image."
            )