# Plant Disease Prediction using Deep Learning

## Overview

Plant Disease Prediction is a deep learning-based image classification project that identifies diseases from plant leaf images. The model is built using **Transfer Learning** with **TensorFlow/Keras** and is designed to help detect plant diseases at an early stage. The project also includes a **Streamlit** web application for easy image upload and real-time prediction.

---

## Features

* Plant leaf disease classification using Transfer Learning
* Image preprocessing and augmentation
* Model training and fine-tuning
* Model evaluation using standard classification metrics
* Single image prediction
* Interactive Streamlit web application
* Simple and deployment-friendly project structure

---

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* OpenCV
* Pillow
* Matplotlib
* Scikit-learn
* Streamlit

---

## Project Structure

```text
plant-disease-prediction/
│
├── data/
│   ├── train/
│   ├── valid/
│   └── test/
│
├── models/
│   └── best_model.keras
│
├── src/
│   ├── config.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

---

## Dataset

This project uses the **PlantVillage** dataset containing images of healthy and diseased plant leaves.

Expected dataset structure:

```text
data/
├── train/
├── valid/
└── test/
```

---

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd plant-disease-prediction
```

### 2. Create a Virtual Environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Train the Model

```bash
python src/train.py
```

The trained model will be saved inside the `models/` directory.

---

## Evaluate the Model

```bash
python src/evaluate.py
```

Evaluation includes:

* Test Accuracy
* Test Loss
* Classification Report
* Confusion Matrix

---

## Predict a Single Image

```bash
python src/predict.py
```

The prediction script returns:

* Predicted disease
* Confidence score

---

## Run the Streamlit Application

```bash
streamlit run app.py
```

Then open the local URL displayed in the terminal.

---

## Project Workflow

```text
Dataset
   │
   ▼
Data Loading
   │
   ▼
Transfer Learning
   │
   ▼
Model Training
   │
   ▼
Model Evaluation
   │
   ▼
Model Saving
   │
   ▼
Streamlit Deployment
```

---

## Future Improvements

* Support additional plant species
* Display disease descriptions and treatment recommendations
* Add Grad-CAM visualizations for model interpretability
* Deploy using Docker
* Convert the model to TensorFlow Lite for mobile applications

---

## Author

**Afaq Ahmad Khan**
