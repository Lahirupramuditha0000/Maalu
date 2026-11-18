
# MAALU - Fish Species Classification

MAALU is a Deep Learning-powered fish species classification application built using Python, TensorFlow, and Streamlit.

The application allows users to upload fish images and receive predictions from a trained image classification model, along with confidence scores.

## Features

- Fish species image classification
- Deep Learning-based image recognition
- Transfer Learning
- Interactive Streamlit interface
- Prediction confidence visualization
- Image upload and real-time inference

## Technologies Used

- Python
- TensorFlow / Keras
- Convolutional Neural Networks (CNN)
- Transfer Learning
- Streamlit
- NumPy
- Pillow
- Matplotlib

## Project Structure

```text
MAALU/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   └── fish_classifier.keras
│
└── screenshots/
    ├── dashboard.png
    ├── prediction.png
    └── training_curves.png
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/MAALU.git
cd MAALU
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

## Model Information

The application uses a trained deep learning image classification model developed using TensorFlow/Keras.

The model was trained using a transfer learning approach to recognize fish species from input images.

## Application Preview

Screenshots will be added here.

## Future Improvements

- Improve classification accuracy
- Expand the number of fish species
- Add more evaluation metrics
- Explore model optimization
- Deploy the application online

## Author

Lahiru Pramuditha

---

This project was developed as part of my journey into Deep Learning and Computer Vision.
