import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Mall Customer Segmentation")

st.title("🛍️ Mall Customer Segmentation")
st.write("Hierarchical Clustering")

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "hierarchical_mall_customer.pkl")

model = joblib.load(MODEL_PATH)

uploaded_file = st.file_uploader("Upload Mall Customers CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Drop unwanted columns
    df = df.drop(columns=["CustomerID", "Genre", "Gender"], errors="ignore")

    # Select ONLY numeric columns
    X = df.select_dtypes(include=["int64", "float64"])

    clusters = model.fit_predict(X)
    df["Cluster"] = clusters

    st.subheader("Clustered Data")
    st.dataframe(df)

    fig, ax = plt.subplots()
    ax.scatter(
        df["Annual Income (k$)"],
        df["Spending Score (1-100)"],
        c=df["Cluster"]
    )
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score")
    st.pyplot(fig)

