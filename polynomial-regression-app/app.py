import streamlit as st
import numpy as np
import pickle
import os
from sklearn.preprocessing import PolynomialFeatures

st.title("📈 Polynomial Regression App")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "polynomial_model.pkl")

st.write("Looking for model at:", MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found!")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

poly = PolynomialFeatures(degree=2)

x = st.number_input("Enter X value", value=5.0)

if st.button("Predict"):
    x_poly = poly.fit_transform([[x]])
    y_pred = model.predict(x_poly)
    st.success(f"Predicted Y value: {y_pred[0]:.2f}")
