import streamlit as st
import pickle
import numpy as np

st.title("🔵 K-Means Clustering App")

model = pickle.load(open("kmeans_model.pkl", "rb"))

x1 = st.number_input("Feature 1")
x2 = st.number_input("Feature 2")

if st.button("Predict Cluster"):
    cluster = model.predict([[x1, x2]])
    st.success(f"Predicted Cluster: {cluster[0]}")
