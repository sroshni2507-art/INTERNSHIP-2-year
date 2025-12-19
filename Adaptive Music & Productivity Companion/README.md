# Adaptive Music & Productivity Companion

## Description
A Streamlit web app that recommends **tasks and music** based on your **mood, activity, time of day, and goal**. Uses **Naive Bayes** for task prediction and **KNN** for music recommendation.

## Features
- Personalized task & music recommendation
- Playlist link suggestions
- Mood vs Task heatmap visualization
- Adaptive ML models with `.pkl` files

## Tech Stack
- Python, Streamlit
- ML: KNN + Naive Bayes
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
