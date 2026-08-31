# 🎯 Smart Object Detection System

<div align="center">

### Real-Time Object Detection using YOLO11 and Streamlit

Upload an image or video, detect objects using a pre-trained YOLO11 model, visualize bounding boxes with confidence scores, and download the processed results.

<br>

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![YOLO](https://img.shields.io/badge/YOLO-YOLO11-orange?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?style=for-the-badge&logo=streamlit)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-black?style=for-the-badge)

</div>

---

# 📌 Overview

Object Detection is one of the most important applications of Computer Vision. Unlike traditional image classification, object detection not only identifies **what objects are present** in an image but also determines **where those objects are located**.

This project implements a complete **Smart Object Detection System** using a pre-trained **YOLO11 model from Ultralytics**.

The application supports both:

- 🖼️ Image Object Detection
- 🎥 Video Object Detection

Users can upload media directly through an interactive Streamlit interface, run inference using YOLO11, visualize detected objects with colored bounding boxes, view confidence scores, and download the processed output.

---

# 🚀 Features

## 🖼️ Image Detection

- Upload custom images
- Detect multiple objects simultaneously
- Display bounding boxes
- Show object class labels
- Display confidence scores
- Use different colors for different object classes
- Adjust confidence threshold
- Download annotated detection results

## 🎥 Video Detection

- Upload custom videos
- Run frame-by-frame object detection
- Display input and processed videos side-by-side
- Preserve object annotations across frames
- Show bounding boxes and confidence scores
- Generate downloadable output videos

## ⚙️ Detection Controls

The Streamlit interface provides configurable settings such as:

- Confidence Threshold
- YOLO Model Selection
- Image / Video Input Mode

---

# 🧠 What is Object Detection?

Object Detection is a Computer Vision task that identifies objects in an image and determines their location.

For every detected object, the model typically predicts:

1. **Class Label**
2. **Bounding Box**
3. **Confidence Score**

For example:

```text
Person → 92%
Bounding Box → (x1, y1, x2, y2)

