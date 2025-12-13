import streamlit as st
import pickle
import numpy as np

import os
import pickle

MODEL_PATH = os.path.join(os.path.dirname(__file__), "diabetes_model.pkl")
model = pickle.load(open(MODEL_PATH, "rb"))

st.title("Diabetes Prediction using Logistic Regression")

age = st.number_input("Age", min_value=1, max_value=120)
mass = st.number_input("Body Mass Index (BMI)", min_value=0.0)
insu = st.number_input("Insulin Level", min_value=0.0)
plas = st.number_input("Plasma Glucose", min_value=0.0)

if st.button("Predict"):
    input_data = np.array([[age, mass, insu, plas]])
    prediction = model.predict(input_data)

    if prediction[0] == "tested_positive":
        st.error("Person is Diabetic")
    else:
        st.success("Person is Not Diabetic")
