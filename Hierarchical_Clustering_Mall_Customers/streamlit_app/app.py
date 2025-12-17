import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

st.title("Hierarchical Clustering - Mall Customers")

# Upload CSV
uploaded_file = st.file_uploader("Upload Mall Customers CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview")
    st.dataframe(df.head())

    # Drop CustomerID
    df = df.drop("CustomerID", axis=1)

    # Encode Genre
    le = LabelEncoder()
    df["Genre"] = le.fit_transform(df["Genre"])

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # Dendrogram
    st.subheader("Dendrogram")
    fig, ax = plt.subplots()
    dendrogram(linkage(X_scaled, method="ward"))
    st.pyplot(fig)

    # Clustering
    hc = AgglomerativeClustering(n_clusters=5, linkage="ward")
    df["Cluster"] = hc.fit_predict(X_scaled)

    st.subheader("Clustered Data")
    st.dataframe(df)
