# yolov8_streamlit_app.py

import streamlit as st
import cv2
from PIL import Image
import tempfile
from ultralytics import YOLO
import numpy as np
import os

# Load YOLOv8 model (auto-download if not present)
model = YOLO("yolov8n.pt")

from ultralytics import YOLO
try:
    model = YOLO("yolov8n.pt")
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")


# Load class labels from coco.names or fallback to default names
try:
    with open("coco.names", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    class_names = model.model.names  # fallback to internal class names

# Streamlit UI setup
st.set_page_config(page_title="YOLOv8 Object Detection", layout="centered")
st.title("🧠 YOLOv8 Object Detection App")
st.sidebar.title("Choose Input Source")

source_type = st.sidebar.radio("Select Source", ["Image", "Video", "Webcam"])

def draw_results(image, results):
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0])
            label = f"{class_names[cls]} {conf:.2f}"

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return image

if source_type == "Image":
    uploaded_img = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_img is not None:
        img = Image.open(uploaded_img)
        img_array = np.array(img)
        results = model.predict(img_array)
        annotated = draw_results(img_array.copy(), results)
        st.image(annotated, caption="Detected Image", use_container_width=True)

elif source_type == "Video":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(frame)
            annotated = draw_results(frame.copy(), results)
            stframe.image(annotated, channels="BGR", use_container_width=True)
        cap.release()

elif source_type == "Webcam":
    st.warning("Click 'Start' and allow webcam access.")
    run = st.checkbox('Start Webcam')
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read from webcam.")
            break
        results = model.predict(frame)
        annotated = draw_results(frame.copy(), results)
        FRAME_WINDOW.image(annotated, channels="BGR")

    cap.release()
