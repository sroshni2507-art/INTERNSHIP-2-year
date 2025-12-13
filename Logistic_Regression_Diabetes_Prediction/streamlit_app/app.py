import streamlit as st
import pickle
import numpy as np
import os

# Load model safely
MODEL_PATH = os.path.join(os.path.dirname(__file__), "diabetes_model.pkl")
model = pickle.load(open(MODEL_PATH, "rb"))

st.title("Diabetes Prediction (Logistic Regression)")

age = st.number_input("Age", 0, 120)
mass = st.number_input("BMI", 0.0)
insu = st.number_input("Insulin", 0.0)
plas = st.number_input("Glucose", 0.0)

if st.button("Predict"):
    data = np.array([[age, mass, insu, plas]])
    result = model.predict(data)

    if result[0] == 1:
        st.error("Person is Diabetic ❌")
    else:
        st.success("Person is Not Diabetic ✅")
