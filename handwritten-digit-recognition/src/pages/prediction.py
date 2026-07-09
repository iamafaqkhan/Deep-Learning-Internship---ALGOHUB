import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

from src.predict import (
    load_trained_model,
    preprocess_uploaded_image,
    predict_digit
)


@st.cache_resource
def load_model():
    return load_trained_model()


model = load_model()


def prediction_page():

    st.title("Digit Prediction")

    uploaded_file = st.file_uploader(
        "Upload a handwritten digit",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Image",
            width=250
        )

        processed = preprocess_uploaded_image(image)

        digit, confidence, probabilities = predict_digit(
            model,
            processed
        )

        st.success(f"Predicted Digit : {digit}")

        st.info(f"Confidence : {confidence*100:.2f}%")

        fig, ax = plt.subplots(figsize=(8,4))

        ax.bar(range(10), probabilities)

        ax.set_xlabel("Digits")
        ax.set_ylabel("Probability")

        st.pyplot(fig)