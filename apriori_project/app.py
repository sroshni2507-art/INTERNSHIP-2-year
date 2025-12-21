import streamlit as st
import pickle
import pandas as pd
import os

# App Title
st.set_page_config(page_title="Apriori Algorithm", layout="centered")
st.title("🛒 Market Basket Analysis using Apriori")

# Get current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# PKL file path
PKL_PATH = os.path.join(BASE_DIR, "apriori_rules.pkl")

# Load PKL file
with open(PKL_PATH, "rb") as file:
    rules = pickle.load(file)

st.success("Apriori rules loaded successfully!")

# Convert frozenset to string (for display)
rules['antecedents'] = rules['antecedents'].apply(
    lambda x: ', '.join(list(x))
)
rules['consequents'] = rules['consequents'].apply(
    lambda x: ', '.join(list(x))
)

# Sidebar filter
st.sidebar.header("Filter Rules")
min_conf = st.sidebar.slider("Minimum Confidence", 0.0, 1.0, 0.4)
min_lift = st.sidebar.slider("Minimum Lift", 0.0, 5.0, 1.0)

filtered_rules = rules[
    (rules['confidence'] >= min_conf) &
    (rules['lift'] >= min_lift)
]

# Display results
st.subheader("📊 Association Rules")
st.dataframe(
    filtered_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
)

st.info(f"Total Rules Shown: {len(filtered_rules)}")
