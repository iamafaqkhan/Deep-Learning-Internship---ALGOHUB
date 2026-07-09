import streamlit as st

from src.pages.home import home_page
from src.pages.prediction import prediction_page
from src.pages.about import about_page

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="",
    layout="wide"
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "Home",
        "Prediction",
        "About"
    ]
)

if page == "Home":
    home_page()

elif page == "Prediction":
    prediction_page()

else:
    about_page()