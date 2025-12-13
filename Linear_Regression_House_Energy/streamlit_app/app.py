import streamlit as st
import pickle
import numpy as np
import os

# Page title
st.set_page_config(page_title="House Energy Prediction", layout="centered")
st.title("🏠 House Energy Prediction App")
st.write("Predict Sub Metering 3 based on Global Active Power using Linear Regression.")

# Load trained model
model_path = os.path.join(os.path.dirname(__file__), "houseenergy_model.pkl")
with open(model_path, "rb") as file:
    model = pickle.load(file)

# ✅ ONLY ONE INPUT (same as training)
global_active_power = st.number_input(
    "Enter Global Active Power",
    min_value=0.0,
    step=0.1
)

# Predict button
if st.button("Predict Energy"):
    input_data = np.array([[global_active_power]])  # ✅ 1 feature only
    prediction = model.predict(input_data)
    st.success(f"🔋 Predicted Sub Metering 3: {prediction[0]:.2f}")
