# Object-Detection-Yolo
This is a Streamlit web application for real-time object detection using the YOLOv8 model by Ultralytics. It supports image, video, and webcam inputs for detecting objects from the COCO dataset.

# 📸 Features:
- **📷 Image Detection:** Upload images (JPG, JPEG, PNG) to detect objects with bounding boxes and confidence scores.
- **🎞️ Video Detection:** Upload videos ( MP4 ) for frame-by-frame object detection, displayed in real time.
- **🎥 Webcam Detection (Live):** Use a webcam for live object detection, with results streamed directly in the app.
- **📜 COCO Class Support:** Detects objects from the COCO dataset's 80 classes, with labels loaded from `coco.names`.
 
# 🧰 Technologies Used:
- **Python:** Core programming language for the application.
- **Streamlit:** A Python library for creating interactive web applications.
- **Ultralytics YOLOv8:** A state-of-the-art object detection model for performing real-time object detection.
- **OpenCV (cv2):** For image and video processing, including drawing bounding boxes and text on detected objects.
- **Pillow (PIL):** For handling image uploads and conversions.

# User interface:
- **YOLOv8 object Detection App** allowing user to choose image, video, or webcam as input source.
- User can select input method through a clear sidebar menu.
- The interface appear clean, running locally via localhost:8501 in a web browser.
![Alt text](https://github.com/farhankhan1112/Object-Detection-Yolo/blob/bf2192e675db461394a3613983735f135e376991/project%20demo%20screenshot/user%20interface.png
)

# Image detection output: 
- Successfully identify object like person, dog, or horse.
- With bounding boxes and confidence scores.
- Upload images (JPG, JPEG, PNG) to detect objects.
![Alt text](https://github.com/farhankhan1112/Object-Detection-Yolo/blob/bf2192e675db461394a3613983735f135e376991/project%20demo%20screenshot/Image%20detection%20output.png
)

# Video deection output: 
- Successfully identify object like moving cars.
- With bounding boxes and confidence scores.
- Upload videos ( MP4 ) for frame-by-frame object detection
![Alt text](https://github.com/farhankhan1112/Object-Detection-Yolo/blob/bf2192e675db461394a3613983735f135e376991/project%20demo%20screenshot/Video%20detection%20output.png
) 

# Webcam detection output: 
- Identify live object through webcam.
- Use a webcam for live object detection.
![Alt text](https://github.com/farhankhan1112/Object-Detection-Yolo/blob/bf2192e675db461394a3613983735f135e376991/project%20demo%20screenshot/Webcame%20detection%20output.png)

# The requirements.txt should include:
- streamlit
- opencv-python-headless
- pillow
- numpy
- torch
- ultralytics

# Acknowledgments:
- Ultralytics for the YOLOv8 model and documentation.
- Streamlit for the web app framework.
- COCO Dataset for the class labels used in object detection.

# License:
- This project is licensed under the MIT License. See the LICENSE file for details.
