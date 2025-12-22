import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from PIL import Image
import os

# Page config
st.set_page_config(
    page_title="Study vs Distraction Environment",
    page_icon="📘",
    layout="centered"
)

st.title("📘 Study vs Distraction Environment Detection")
st.write("Upload an image to classify the environment")

# -------- Model Loader (AUTO detect .h5 or .pkl) --------
@st.cache_resource
def load_model():
    if os.path.exists("study_distraction_model.h5"):
        st.info("Loaded TensorFlow model (.h5)")
        return tf.keras.models.load_model("study_distraction_model.h5"), "h5"

    elif os.path.exists("model.pkl"):
        st.info("Loaded Pickle model (.pkl)")
        with open("model.pkl", "rb") as f:
            return pickle.load(f), "pkl"

    else:
        st.error("❌ No model file found (.h5 or .pkl)")
        return None, None

model, model_type = load_model()

# -------- Image Upload --------
uploaded_file = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image (common for both)
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    if model_type == "h5":
        prediction = model.predict(img)

    else:  # pkl model
        prediction = model.predict_proba(img.reshape(1, -1))[:, 1]

    st.subheader("🔍 Prediction Result")

    if prediction[0] > 0.5:
        st.success("📘 Study Environment")
    else:
        st.error("📵 Distraction Environment")
