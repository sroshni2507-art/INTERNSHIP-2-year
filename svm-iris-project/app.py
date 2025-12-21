import streamlit as st
import pickle
import numpy as np
import os
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

st.title("🌸 Iris Flower Classification – SVM")

# -------- SAFE MODEL PATH --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "svm_model.pkl")

# -------- LOAD MODEL --------
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# -------- LOAD DATA FOR SCALER --------
iris = load_iris()
scaler = StandardScaler()
scaler.fit(iris.data)

# -------- USER INPUT --------
sepal_length = st.number_input("Sepal Length", 4.0, 8.0, 5.1)
sepal_width  = st.number_input("Sepal Width", 2.0, 4.5, 3.5)
petal_length = st.number_input("Petal Length", 1.0, 7.0, 1.4)
petal_width  = st.number_input("Petal Width", 0.1, 2.5, 0.2)

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    st.success(f"🌼 Predicted Species: {iris.target_names[prediction][0]}")

