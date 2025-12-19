import streamlit as st
import numpy as np
import pickle
import os
from sklearn.preprocessing import PolynomialFeatures

st.title("📈 Polynomial Regression App")

# ---- Safe path ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "polynomial_model.pkl")

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found!")
    st.stop()

# ---- Load model ----
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

poly = PolynomialFeatures(degree=2)

# ---- User input ----
x_value = st.number_input("Enter X value", min_value=0.0, value=5.0)

if st.button("Predict"):
    x_array = np.array([[x_value]])
    x_poly = poly.fit_transform(x_array)
    y_pred = model.predict(x_poly)

    st.success(f"Predicted Y value: {y_pred[0]:.2f}")
