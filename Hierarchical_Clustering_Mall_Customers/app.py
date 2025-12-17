import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load model & scaler
BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, "hierarchical_mall_customer.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

st.title("🛍️ Mall Customer Segmentation")
st.write("Hierarchical Clustering using Saved Model")

uploaded_file = st.file_uploader("Upload Mall Customers CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -------------------------
    # Preprocessing
    # -------------------------
    if "CustomerID" in df.columns:
        df.drop("CustomerID", axis=1, inplace=True)

    if "Gender" in df.columns:
        le = LabelEncoder()
        df["Gender"] = le.fit_transform(df["Gender"])

    # ✅ IMPORTANT: select same features used during training
    feature_cols = [
        "Gender",
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]

    X = df[feature_cols]

    # Scaling
    X_scaled = scaler.transform(X)

    # Clustering
    clusters = model.fit_predict(X_scaled)
    df["Cluster"] = clusters

    st.subheader("Clustered Data")
    st.dataframe(df.head())

    # Visualization
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

else:
    st.info("Please upload the Mall Customers CSV file.")
