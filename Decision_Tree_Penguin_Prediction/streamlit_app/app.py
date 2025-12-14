import streamlit as st
import pickle
import numpy as np
import os

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "../model/penguin_dtree_model.pkl"
)

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🐧 Penguin Species Prediction")
st.write("Decision Tree Model")

# Input fields (MATCH NOTEBOOK)
culmen_length = st.number_input("Culmen Length (mm)", 0.0)
culmen_depth = st.number_input("Culmen Depth (mm)", 0.0)
flipper_length = st.number_input("Flipper Length (mm)", 0.0)
body_mass = st.number_input("Body Mass (g)", 0.0)
sex = st.selectbox("Sex", ["Male", "Female"])

# Encode sex (same as notebook)
sex_val = 1 if sex == "Male" else 0

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Species"):
    input_data = np.array([[
        culmen_length,
        culmen_depth,
        flipper_length,
        body_mass,
        sex_val
    ]])

    prediction = model.predict(input_data)

    species_map = {
        0: "Adelie",
        1: "Chinstrap",
        2: "Gentoo"
    }

    st.success(f"🧬 Predicted Species: **{species_map[prediction[0]]}**")
