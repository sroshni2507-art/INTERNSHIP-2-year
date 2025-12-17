import streamlit as st
import cv2
import face_recognition
import pickle
import numpy as np
from PIL import Image

st.set_page_config(page_title="Face Recognition App", layout="centered")

st.title("😀 Face Recognition using Streamlit")
st.write("Upload an image or use webcam to recognize faces")

# Load PKL file
with open("face_encodings.pkl", "rb") as f:
    data = pickle.load(f)

option = st.radio("Choose Input Type", ["Upload Image", "Use Webcam"])

# ---------------- IMAGE UPLOAD ---------------- #
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        face_locations = face_recognition.face_locations(img_array)
        face_encodings = face_recognition.face_encodings(img_array, face_locations)

        for encoding, location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            name = "Unknown"

            if True in matches:
                index = matches.index(True)
                name = data["names"][index]

            top, right, bottom, left = location
            cv2.rectangle(img_array, (left, top), (right, bottom), (0,255,0), 2)
            cv2.putText(img_array, name, (left, top-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        st.image(img_array, caption="Result", use_container_width=True)

# ---------------- WEBCAM ---------------- #
elif option == "Use Webcam":
    run = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.error("Webcam not working")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb)
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        for encoding, location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            name = "Unknown"

            if True in matches:
                index = matches.index(True)
                name = data["names"][index]

            top, right, bottom, left = location
            cv2.rectangle(rgb, (left, top), (right, bottom), (0,255,0), 2)
            cv2.putText(rgb, name, (left, top-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        FRAME_WINDOW.image(rgb)

    camera.release()
