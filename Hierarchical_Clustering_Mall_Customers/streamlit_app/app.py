import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Load model & scaler safely
# -----------------------------
BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "hierarchical_mall_customer.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🛍️ Mall Customer Segmentation")
st.write("Hierarchical Clustering using Saved .pkl Model")

uploaded_file = st.file_uploader(
    "Upload Mall Customers CSV",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------
    # Preprocessing
    # -----------------------------
    if "CustomerID" in df.columns:
        df = df.drop("CustomerID", axis=1)

    if "Gender" in df.columns:
        le = LabelEncoder()
        df["Gender"] = le.fit_transform(df["Gender"])

    # Scaling
    X = scaler.transform(df)

    # -----------------------------
    # Clustering
    # -----------------------------
    clusters = model.fit_predict(X)
    df["Cluster"] = clusters

    st.subheader("Clustered Data")
    st.dataframe(df)

    # -----------------------------
    # Visualization
    # -----------------------------
    st.subheader("Cluster Visualization")

    plt.figure(figsize=(6, 4))
    plt.scatter(
        df["Annual Income (k$)"],
        df["Spending Score (1-100)"],
        c=df["Cluster"]
    )
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.title("Hierarchical Clustering Output")
    st.pyplot(plt)

else:
    st.info("Please upload the Mall Customers CSV file.")
