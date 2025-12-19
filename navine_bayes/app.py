import streamlit as st
import pickle
import pandas as pd
import os

# -------------------- LOAD MODEL SAFELY --------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "naive_bayes_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# -------------------- PAGE SETUP --------------------
st.set_page_config(page_title="Naive Bayes Predictor", layout="centered")
st.title("Naive Bayes Predictor 🧠")

# -------------------- GET FEATURE NAMES FROM MODEL --------------------
try:
    feature_names = model.feature_names_in_
except AttributeError:
    st.error("❌ This model was trained without feature names.")
    st.stop()

# -------------------- USER INPUTS --------------------
st.subheader("Enter Input Values")

user_inputs = {}
for feature in feature_names:
    user_inputs[feature] = st.text_input(f"{feature}")

# -------------------- PREDICTION --------------------
if st.button("Predict"):
    input_df = pd.DataFrame([user_inputs])

    try:
        prediction = model.predict(input_df)
        st.success(f"Prediction: {prediction[0]}")
    except Exception as e:
        st.error("Prediction failed. Check input format.")
        st.exception(e)

