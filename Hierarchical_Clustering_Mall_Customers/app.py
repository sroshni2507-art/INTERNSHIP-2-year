import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "hierarchical_mall_customer.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

st.title("🛍️ Mall Customer Segmentation (Hierarchical Clustering)")

uploaded_file = st.file_uploader("Upload Mall Customers CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # 🔹 Drop unwanted columns
    df = df.drop(columns=["CustomerID"], errors="ignore")

    # 🔹 Rename Genre → Gender
    df.rename(columns={"Genre": "Gender"}, inplace=True)

    # 🔹 FORCE EXACT TRAINING COLUMN NAMES
    df = df.rename(columns={
        "Age": "Age",
        "Annual Income (k$)": "Annual Income (k$)",
        "Spending Score (1-100)": "Spending Score (1-100)"
    })

    # 🔹 SELECT EXACT FEATURES USED DURING TRAINING
    X = df[[
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]]

    # 🔹 Scale
    X_scaled = scaler.transform(X)

    # 🔹 Predict clusters
    df["Cluster"] = model.fit_predict(X_scaled)

    st.subheader("Clustered Output")
    st.dataframe(df.head())

    # 🔹 Visualization
    st.subheader("Customer Segmentation Plot")
    plt.figure(figsize=(6, 4))
    plt.scatter(
        df["Annual Income (k$)"],
        df["Spending Score (1-100)"],
        c=df["Cluster"]
    )
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    st.pyplot(plt)
