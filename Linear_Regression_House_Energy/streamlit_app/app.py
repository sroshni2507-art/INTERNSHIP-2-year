import streamlit as st
import pickle
import numpy as np
import os

# Page title
st.set_page_config(page_title="House Energy Prediction", layout="centered")
st.title("🏠 House Energy Prediction App")
st.write("Predict house energy usage based on input features using Linear Regression.")

# Load trained model
model_path = os.path.join(os.path.dirname(__file__), "../houseenergy_model(1).pkl")
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Example input fields - adjust based on your model features
feature1 = st.number_input("Enter Feature 1", min_value=0.0)
feature2 = st.number_input("Enter Feature 2", min_value=0.0)
# Add more features if your model needs

# Predict button
if st.button("Predict Energy"):
    input_data = np.array([[feature1, feature2]])  # Make sure this matches model features
    prediction = model.predict(input_data)
    st.success(f"🔋 Predicted Energy Consumption: {prediction[0]:,.2f}")
