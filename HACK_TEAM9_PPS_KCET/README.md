# 📘 Study vs Distraction Environment Detection


## 📦 Large Files (Model / Dataset)

Due to GitHub file size limitations (25 MB), the large files used in this project are hosted on Google Drive.

🔗 **Download Link**:  
https://drive.google.com/drive/folders/1pIGlP8iJjPgxk9VdCqZ69qEj7t5_2fUz?usp=drive_link

📌 Please download the files from the above link and place them in the required project folders before running the code.


## 🔍 Problem Statement
Students often face distractions while studying.
This project aims to classify an image as either a **Study Environment** or a **Distraction Environment** using Machine Learning / Deep Learning techniques.

---

## 🎯 Objective
To build and deploy an image classification model that predicts:
- 📘 Study Environment  
- 📵 Distraction Environment  

using uploaded images.

---

## 📊 Dataset Collection
- Dataset was **self-collected using Google Forms**
- Participants uploaded real-time images
- Each image was labeled as:
  - Study Environment
  - Distraction Environment
- This ensures originality and real-world relevance

📎 Google Form link is provided in `form_link.txt`

---

## 🧠 Models Used
This project supports **two types of trained models**:

### 1️⃣ CNN Model (`.h5`)
- Built using **TensorFlow / Keras**
- Used for image-based deep learning
- Preferred model for deployment

### 2️⃣ Pickle Model (`.pkl`)
- Serialized using Python Pickle
- Used for demonstration / traditional ML compatibility

👉 The Streamlit app automatically loads:
- `.h5` model if available  
- Otherwise `.pkl` model

---

## 🛠️ Technologies Used
- Python  
- TensorFlow / Keras  
- Scikit-learn  
- Google Colab  
- Streamlit  
- GitHub  

---

## 🚀 How to Run the Project

### ▶️ Step 1: Clone the Repository
```bash
git clone <your-github-repo-link>
cd HACK_TEAM_PPS_KCET
