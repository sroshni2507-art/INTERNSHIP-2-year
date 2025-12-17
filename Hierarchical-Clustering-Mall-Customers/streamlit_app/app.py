import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("hierarchical_mall_customer(1).pkl")
scaler = joblib.load("scaler.pkl")

st.title("🛍️ Mall Customer Segmentation")
st.write("Hierarchical Clustering (Unsupervised Learning)")

# Upload dataset
uploaded_file = st.file_uploader("Upload Mall Customer CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Preprocessing
    df = df.drop("CustomerID", axis=1)
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

    X = scaler.transform(df)

    # Clustering
    clusters = model.fit_predict(X)
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
    plt.xlabel("Annual Income")
    plt.ylabel("Spending Score")
    st.pyplot(plt)
