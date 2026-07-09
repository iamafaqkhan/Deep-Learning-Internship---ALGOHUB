import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from streamlit_drawable_canvas import st_canvas

from src.predict import (
    load_trained_model,
    preprocess_uploaded_image,
    predict_digit
)


@st.cache_resource
def load_model():
    return load_trained_model()


model = load_model()


def show_prediction(image, caption):

    st.image(image, caption=caption, width=250)

    processed = preprocess_uploaded_image(image)

    digit, confidence, probabilities = predict_digit(
        model,
        processed
    )

    st.success(f"Predicted Digit: {digit}")

    st.info(f"Confidence: {confidence * 100:.2f}%")

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(range(10), probabilities)

    ax.set_xticks(range(10))
    ax.set_xlabel("Digits")
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Probabilities")

    st.pyplot(fig)


def prediction_page():

    st.title("Handwritten Digit Recognition")

    tab1, tab2 = st.tabs(["Upload Image", "Draw Digit"])

    # -----------------------------
    # Upload Image
    # -----------------------------
    with tab1:

        uploaded_file = st.file_uploader(
            "Upload a handwritten digit",
            type=["png", "jpg", "jpeg"],
            key="upload"
        )

        if uploaded_file is not None:

            image = Image.open(uploaded_file)

            show_prediction(
                image,
                "Uploaded Image"
            )

    # -----------------------------
    # Draw Digit
    # -----------------------------
    with tab2:

        st.write("Draw a white digit on the black canvas.")

        canvas_result = st_canvas(

            fill_color="black",

            stroke_width=18,

            stroke_color="white",

            background_color="black",

            width=280,

            height=280,

            drawing_mode="freedraw",

            key="canvas",
        )

        if canvas_result.image_data is not None:

            image = Image.fromarray(
                canvas_result.image_data.astype(np.uint8)
            )

            if st.button("Predict Drawn Digit"):

                show_prediction(
                    image,
                    "Drawn Digit"
                )