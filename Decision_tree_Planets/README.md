# 🪐 Planets Dataset Analysis (Google Colab Project)

## 📌 Project Title

**Exploratory Data Analysis of Exoplanet Discoveries using Seaborn Planets Dataset**

---

## 🎯 Objective

The objective of this project is to explore and analyze the **Planets (Exoplanets) dataset** using Python and Seaborn. The project focuses on understanding how planets are discovered, trends over the years, and relationships between different planetary features.

---

## 📂 Dataset Information

* **Dataset Name:** Planets
* **Source:** Seaborn built-in dataset
* **Dataset Type:** Structured tabular data
* **Rows:** 1035
* **Columns:** 6

### Columns Description:

* **method** – Planet discovery method (Transit, Radial Velocity, etc.)
* **year** – Year of discovery
* **orbital_period** – Orbital period of the planet
* **mass** – Mass of the planet
* **distance** – Distance from Earth
* **number** – Number of planets discovered

---

## 🛠️ Tools & Libraries Used

* Python
* Google Colab
* Pandas
* Seaborn
* Matplotlib

---

## 📊 Analysis & Visualizations

The following visualizations are performed in this project:

1. **Planets discovered per year** (Line Plot)
2. **Discovery method count** (Bar Chart)
3. **Orbital period vs Mass** (Scatter Plot)
4. **Mass distribution of planets** (Histogram)
5. **Year vs Discovery Method** (Heatmap)

These plots help in understanding trends and patterns in exoplanet discoveries.

---

## ✅ Key Insights

* Planet discoveries increased significantly after the year 2000.
* Transit and Radial Velocity are the most commonly used discovery methods.
* There is a wide variation in orbital periods and planet masses.
* Certain discovery methods dominate in specific years.

---

## ▶️ How to Run the Project

1. Open **Google Colab**
2. Install required libraries (if needed)
3. Run the following code to load the dataset:

```python
import seaborn as sns
df = sns.load_dataset("planets")
```

4. Execute the analysis and visualization cells

---

## 📁 Project Structure

```
planets-analysis/
│
├── planets_analysis.ipynb
├── README.md
└── requirements.txt
```

---

## 📌 Conclusion

This project provides a clear understanding of **exoplanet discovery trends** using visual analytics. It is a beginner-friendly project suitable for learning **EDA, Seaborn visualizations, and data interpretation**.

---

## 👩‍💻 Author

**Name:** Your Name Here
**Course:** Machine Learning / Data Science
**Platform:** Google Colab

---

⭐ *This project is ideal for college mini-projects and ML/Data Science portfolios.*
