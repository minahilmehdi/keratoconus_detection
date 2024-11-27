import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from urllib.request import urlopen

# Load your pre-trained models
svm_model = joblib.load("svm_keratoconus_model.pkl")

categories = ["Keratoconus", "Normal", "Suspect"]

def preprocess_image(img_path):
    """Preprocess image from URL."""
    if img_path.startswith('http'):
        # Open the URL and read it as a byte stream
        resp = urlopen(img_path)
        img_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(img_path)

    if img is None:
        raise ValueError(f"Image not found: {img_path}")

    # Resize the image to (64x64) for HOG extraction
    img = cv2.resize(img, (64, 64))

    # Convert to grayscale for HOG feature extraction
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Extract HOG features
    features, _ = hog(
        gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True
    )

    return features

def make_prediction(image_url):
    """Make prediction using SVM model."""
    features = preprocess_image(image_url).reshape(1, -1)
    prediction = svm_model.predict(features)
    return categories[prediction[0]]

# Streamlit app UI
st.title('Keratoconus Detection Model')
st.write("Enter an image URL below to predict the condition (Keratoconus, Normal, or Suspect).")

# Text input for the image URL
image_url = st.text_input("Enter image URL:")

if image_url:
    try:
        # Display image preview
        st.image(image_url, caption="Input Image", use_column_width=True)

        # Make the prediction
        prediction = make_prediction(image_url)
        st.write(f"Predicted Category: {prediction}")
    except Exception as e:
        st.error(f"Error: {e}")

