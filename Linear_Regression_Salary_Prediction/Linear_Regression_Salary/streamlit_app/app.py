import streamlit as st
import pickle
import numpy as np

# Page title
st.set_page_config(page_title="Salary Prediction App", layout="centered")

st.title("💼 Salary Prediction App")
st.write("This app predicts salary based on years of experience using Linear Regression.")

import os

model_path = os.path.join(os.path.dirname(__file__), "salary_model.pkl")
with open(model_path, "rb") as file:
    model = pickle.load(file)

years_exp = st.number_input(
    "Enter Years of Experience",
    min_value=0.0,
    max_value=50.0,
    step=0.5
)

# Predict button
if st.button("Predict Salary"):
    input_data = np.array([[years_exp]])
    prediction = model.predict(input_data)
    st.success(f"💰 Predicted Salary: ₹ {prediction[0]:,.2f}")
