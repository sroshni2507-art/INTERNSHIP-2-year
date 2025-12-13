import streamlit as st
import pickle
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "diabetes_model.pkl")
model = pickle.load(open(MODEL_PATH, "rb"))

st.title("Diabetes Prediction using Logistic Regression")

preg = st.number_input("Pregnancies", 0)
glucose = st.number_input("Glucose Level", 0)
bp = st.number_input("Blood Pressure", 0)
skin = st.number_input("Skin Thickness", 0)
insulin = st.number_input("Insulin", 0)
bmi = st.number_input("BMI", 0.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0)
age = st.number_input("Age", 0)

if st.button("Predict"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    result = model.predict(input_data)

    if result[0] == 1:
        st.error("Person is Diabetic")
    else:
        st.success("Person is Not Diabetic")
