import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os

st.set_page_config(page_title="Alzheimer Detection", layout="centered")
st.title("🧠 Alzheimer Disease Detection (CNN)")

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "alzheimer_model.h5"
)

# Check model file
if not os.path.exists(MODEL_PATH):
    st.warning(
        "⚠️ Model file not found!\n\n"
        "Please download or train `alzheimer_model.h5` and place it inside "
        "`streamlit_app/` folder."
    )
    st.stop()

# Load model
model = load_model(MODEL_PATH)

uploaded = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = img.reshape(1, 224, 224, 3)

    if st.button("Predict"):
        pred = model.predict(img)[0][0]

        if pred > 0.5:
            st.error("🟥 Alzheimer Detected")
        else:
            st.success("🟩 No Alzheimer Detected")
