import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

st.title("🧠 Brain Tumor Detection (CNN)")

# 🔹 Model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "brain_tumor_model.h5")

# 🔹 Load model safely
if not os.path.exists(MODEL_PATH):
    st.warning("⚠️ Model file not found! Please download 'brain_tumor_model.h5' and place it here.")
else:
    model = load_model(MODEL_PATH)

    # 🔹 File uploader
    uploaded = st.file_uploader("Upload MRI Image", type=["jpg", "png"])

    if uploaded:
        image = Image.open(uploaded).resize((224,224))
        img = np.array(image)/255.0
        img = img.reshape(1,224,224,3)

        # 🔹 Prediction
        pred = model.predict(img)

        if pred[0][0] > 0.5:
            st.error("Tumor Detected ❌")
        else:
            st.success("No Tumor ✅")
