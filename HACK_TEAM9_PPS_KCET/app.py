import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page config
st.set_page_config(
    page_title="Study vs Distraction Environment",
    page_icon="📘",
    layout="centered"
)

st.title("📘 Study vs Distraction Environment Detection")
st.write("Upload an image to classify the environment")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("study_distraction_model.h5")
    return model

model = load_model()

# Image upload
uploaded_file = st.file_uploader(
    "📤 Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)

    st.subheader("🔍 Prediction Result")

    if prediction[0][0] > 0.5:
        st.success("📘 Study Environment")
    else:
        st.error("📵 Distraction Environment")
