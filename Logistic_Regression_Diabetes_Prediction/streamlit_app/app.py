import streamlit as st
import pickle
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "diabetes_model.pkl")
model = pickle.load(open(MODEL_PATH, "rb"))

st.title("Diabetes Prediction using Logistic Regression")

age = st.number_input("Age", 0, 100)
mass = st.number_input("BMI", 0.0)
insu = st.number_input("Insulin", 0.0)
plas = st.number_input("Glucose Level", 0.0)

if st.button("Predict"):
    data = np.array([[age, mass, insu, plas]])
    pred = model.predict(data)

    if pred[0] == 1:
        st.error("Person is Diabetic ❌")
    else:
        st.success("Person is Not Diabetic ✅")
