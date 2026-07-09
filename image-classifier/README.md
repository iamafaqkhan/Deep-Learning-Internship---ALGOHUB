# Project Title: Image Classification

Deep Learning Internship Project – **AlgoHub Software House**

## Project Overview

This project implements an image classification model using Deep Learning to recognize and classify images into predefined categories. The model is trained using TensorFlow/Keras and deployed with Streamlit, allowing users to upload images and receive real-time predictions.

## Technologies

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Streamlit
* Git

## Project Structure

```text
app/                # Streamlit application
data/               # Dataset
notebooks/          # Experiments
src/                # Source code
saved_models/       # Trained models
reports/            # Results and evaluation
tests/              # Test scripts
```

## Installation

```bash
git clone <repository-url>
cd <repository-name>

uv venv
.venv\Scripts\activate

uv pip install -r requirements.txt
```

## Usage

Train the model:

```bash
python src/models/train.py
```

Launch the Streamlit application:

```bash
streamlit run app/app.py
```

## Dataset info

Dataset: CIFAR-10
Total Images: 60,000 RGB images
Training Set: 50,000 images
Test Set: 10,000 images
Classes: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, and Truck.
Each image has a resolution of 32 × 32 pixels. The dataset is widely used as a benchmark for image classification tasks.

## Results

* Model training and evaluation
* Performance metrics
* Prediction results
* Streamlit web interface

## Repository Contents

* Complete source code
* Project documentation
* Dataset information
* Requirements file
* Training results
* Application interface

## Author

**Afaq Ahmad Khan**
**Linkedin:** <https://www.linkedin.com/in/afaqahmadkhan44>

**Project Demo:** <>

Deep Learning Internship – AlgoHub Software House
