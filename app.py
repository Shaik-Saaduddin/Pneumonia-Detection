import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load the model
MODEL_PATH = "pneumonia_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Inject custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("frontend/style.css")

# Top navigation bar
st.markdown("""
    <div class="top-nav">
        <div class="nav-left">
            <img src="https://cdn-icons-png.flaticon.com/512/2927/2927700.png" width="40" />
            <h2>Pneumonia Detection</h2>
        </div>
    </div>
""", unsafe_allow_html=True)

# Page Title
st.markdown("<h1>Pneumonia Detection from Chest X-rays</h1>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Predict"):
        prediction = model.predict(img_array)[0][0]

        if prediction < 0.5:
            st.success("✅ Normal - No Pneumonia Detected")
        else:
            st.error("⚠️ Pneumonia Detected")
