import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "hierarchical_mall_customer.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

st.title("🛍️ Mall Customer Segmentation")

uploaded_file = st.file_uploader("Upload Mall Customers CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Drop ID
    if "CustomerID" in df.columns:
        df.drop("CustomerID", axis=1, inplace=True)

    # Rename Genre → Gender
    if "Genre" in df.columns:
        df.rename(columns={"Genre": "Gender"}, inplace=True)

    # Encode Gender (NOT used for scaling)
    if "Gender" in df.columns:
        df["Gender"] = LabelEncoder().fit_transform(df["Gender"])

    # ✅ ONLY NUMERIC FEATURES USED DURING TRAINING
    X = df[[
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]]

    # Scale
    X_scaled = scaler.transform(X)

    # Clustering
    clusters = model.fit_predict(X_scaled)
    df["Cluster"] = clusters

    st.subheader("Clustered Data")
    st.dataframe(df.head())

    st.subheader("Cluster Visualization")
    plt.figure(figsize=(6, 4))
    plt.scatter(
        df["Annual Income (k$)"],
        df["Spending Score (1-100)"],
        c=df["Cluster"]
    )
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    st.pyplot(plt)
