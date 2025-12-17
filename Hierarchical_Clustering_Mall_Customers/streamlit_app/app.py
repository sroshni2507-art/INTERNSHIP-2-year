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
st.write("Hierarchical Clustering using Saved Model")

uploaded_file = st.file_uploader("Upload Mall Customers CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "CustomerID" in df.columns:
        df.drop("CustomerID", axis=1, inplace=True)

    if "Gender" in df.columns:
        le = LabelEncoder()
        df["Gender"] = le.fit_transform(df["Gender"])

    X = scaler.transform(df)

    clusters = model.fit_predict(X)
    df["Cluster"] = clusters

    st.dataframe(df.head())

    plt.figure()
    plt.scatter(
        df["Annual Income (k$)"],
        df["Spending Score (1-100)"],
        c=df["Cluster"]
    )
    st.pyplot(plt)
