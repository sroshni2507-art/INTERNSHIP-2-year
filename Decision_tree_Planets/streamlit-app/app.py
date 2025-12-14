import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Planets Data Explorer",
    page_icon="🪐",
    layout="centered"
)

st.title("🪐 Planets Dataset Explorer")
st.write("Explore **exoplanet discoveries** using the Seaborn planets dataset.")

# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_data():
    return sns.load_dataset("planets")

df = load_data()

# ==============================
# SIDEBAR FILTERS
# ==============================
st.sidebar.header("🔍 Filters")

method = st.sidebar.multiselect(
    "Discovery Method",
    options=df["method"].unique(),
    default=df["method"].unique()
)

year_range = st.sidebar.slider(
    "Discovery Year",
    int(df["year"].min()),
    int(df["year"].max()),
    (2000, 2015)
)

# Apply filters
filtered_df = df[
    (df["method"].isin(method)) &
    (df["year"] >= year_range[0]) &
    (df["year"] <= year_range[1])
]

# ==============================
# DATA OVERVIEW
# ==============================
st.subheader("📊 Dataset Overview")
st.write(f"Total Records: **{filtered_df.shape[0]}**")
st.dataframe(filtered_df.head())

# ==============================
# PLOT 1: Discoveries by Year
# ==============================
st.subheader("📈 Planets Discovered per Year")

fig1, ax1 = plt.subplots()
filtered_df["year"].value_counts().sort_index().plot(kind="line", ax=ax1)
ax1.set_xlabel("Year")
ax1.set_ylabel("Number of Planets")
st.pyplot(fig1)

# ==============================
# PLOT 2: Discovery Method Count
# ==============================
st.subheader("🔭 Discovery Methods")

fig2, ax2 = plt.subplots()
filtered_df["method"].value_counts().plot(kind="bar", ax=ax2)
ax2.set_xlabel("Method")
ax2.set_ylabel("Count")
st.pyplot(fig2)

# ==============================
# PLOT 3: Orbital Period vs Mass
# ==============================
st.subheader("🌍 Orbital Period vs Mass")

fig3, ax3 = plt.subplots()
ax3.scatter(
    filtered_df["orbital_period"],
    filtered_df["mass"]
)
ax3.set_xlabel("Orbital Period")
ax3.set_ylabel("Mass")
st.pyplot(fig3)

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.markdown("**Streamlit App | Seaborn Planets Dataset | Data Visualization Project**")
