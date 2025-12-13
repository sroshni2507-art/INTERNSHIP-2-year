import streamlit as st
import numpy as np
import os
import pickle

# Construct the file path relative to the repository root.
# On Streamlit Cloud, the working directory is the repo root.
MODEL_PATH = "Decision_Tree_Penguin_Prediction/streamlit_app/penguin_dtree_model.pkl" 

# Load the model
# Ensure you've followed the Git LFS steps mentioned previously for proper upload.
model = pickle.load(open(MODEL_PATH, "rb"))

st.title("Penguin Species Prediction (Decision Tree)")

bill_length = st.number_input("Bill Length (mm)", 0.0)
bill_depth = st.number_input("Bill Depth (mm)", 0.0)
flipper_length = st.number_input("Flipper Length (mm)", 0.0)
body_mass = st.number_input("Body Mass (g)", 0)
sex = st.selectbox("Sex", ["Male", "Female"])
island = st.selectbox("Island", ["Biscoe", "Dream", "Torgersen"])

# Encoding user input same as training
sex_val = 1 if sex == "Male" else 0
island_map = {"Biscoe": 0, "Dream": 1, "Torgersen": 2}
island_val = island_map[island]

if st.button("Predict"):
    data = np.array([[bill_length, bill_depth, flipper_length, body_mass, sex_val, island_val]])
    pred = model.predict(data)
    species_map = {0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}
    st.success(f"Predicted Species: {species_map[pred[0]]}")

