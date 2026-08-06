# Day 15 — Object Detection with YOLOv8 🚀

![Python](https://img.shields.io/badge/Python-3.x-blue)
![YOLO](https://img.shields.io/badge/Model-YOLOv8-green)
![Streamlit](https://img.shields.io/badge/Deployment-Streamlit-red)

Part of the **MLBench Summer Internship Program**.

This module focuses on the fundamentals of **Computer Vision Object Detection**, performing inference using pretrained YOLOv8 models, analyzing detection results, and building an interactive Streamlit-based detection dashboard.

---

# 📌 Overview

| Task       | Object Detection using YOLOv8                   |
| ---------- | ----------------------------------------------- |
| Dataset    | Helmet Detection Dataset — Roboflow Universe    |
| Model Used | YOLOv8 Nano (`yolov8n.pt`)                      |
| Framework  | Ultralytics YOLO                                |
| Input      | Images and Videos                               |
| Output     | Bounding boxes, class labels, confidence scores |
| Deployment | Streamlit Detection Dashboard                   |

---

# 🧠 Concepts Covered

## Object Detection vs Image Classification

| Image Classification                        | Object Detection                               |
| ------------------------------------------- | ---------------------------------------------- |
| Predicts a single class for an entire image | Detects multiple objects inside an image       |
| Does not provide object location            | Provides object locations using bounding boxes |
| Output: Class + confidence                  | Output: Class + confidence + coordinates       |

Example:

**Classification:**

> "This image contains a helmet"

**Object Detection:**

> "Helmet detected at this location with 92% confidence"

---

# 🔍 What is Object Detection?

Object detection is a computer vision task that identifies objects present in an image and determines their exact locations.

A detection model produces:

* Object class
* Confidence score
* Bounding box coordinates

This allows applications such as:

* Safety monitoring
* Autonomous vehicles
* Surveillance systems
* Industrial automation

---

# 🤖 What is YOLO?

YOLO (**You Only Look Once**) is a real-time object detection algorithm that performs detection in a single forward pass through a neural network.

Unlike traditional two-stage detectors, YOLO directly predicts:

* Bounding boxes
* Object classes
* Confidence probabilities

This makes YOLO highly suitable for:

* Real-time image detection
* Video processing
* Edge devices

---

# 📂 Dataset

## Helmet Detection Dataset

**Source:** Roboflow Universe

Dataset format:

* YOLO format
* Images with annotation labels
* Train / Validation / Test split

The dataset contains images used for helmet detection and object detection experimentation.

The dataset is intentionally excluded from GitHub because of its large size.

---

# 🤖 Model Used

## YOLOv8 Nano (`yolov8n.pt`)

The project uses a pretrained YOLOv8 Nano model from Ultralytics.

Reasons for selecting YOLOv8 Nano:

* Lightweight architecture
* Fast inference speed
* Suitable for real-time applications
* Lower computational requirements

The model automatically downloads the pretrained weights during execution if they are not available locally.

---

# 🔎 Detection Pipeline

The project contains two main detection implementations.

---

# 1️⃣ YOLO Practice Script

File:

```
yolo_practice.py
```

This script demonstrates the fundamentals of YOLO inference.

Features:

✅ Load pretrained YOLOv8 model
✅ Perform single image detection
✅ Perform multiple image detection
✅ Display confidence scores
✅ Extract bounding boxes
✅ Display detected class labels
✅ Save prediction results

---

# 2️⃣ Helmet Detection System

File:

```
object_detection.py
```

This implements the complete object detection workflow.

## Pipeline:

### Dataset Loading

Images are loaded from the Helmet Detection dataset.

### Model Inference

YOLOv8 processes images and predicts:

* Objects
* Locations
* Confidence scores

### Visualization

Generated outputs include:

* Bounding boxes
* Class names
* Confidence values

### Result Analysis

The system generates detection summaries including:

* Number of detected objects
* Detected classes
* Confidence information

---

# 📊 Observations

During experimentation:

* YOLOv8 successfully detects objects present in the image based on its learned classes.
* Detection confidence depends on image quality, lighting conditions, object visibility, and size.
* Clear and centered objects generally produce higher confidence scores.
* Small, partially visible, or blurred objects may produce lower confidence predictions.
* Increasing model size can improve accuracy but increases computational cost.

---

# 🌐 Streamlit Detection Dashboard

File:

```
[streamlit_app.py]
https://mlb-internship-amnhhdcmfxel5bxawmkyfu.streamlit.app/
```

The project includes an interactive Streamlit application for real-time object detection.

---

# 🚀 Dashboard Features

## Image Detection

Users can:

✅ Upload an image
✅ Run YOLO detection
✅ View annotated output
✅ Download processed image

---

## Video Detection

Users can:

✅ Upload video files
✅ Perform frame-by-frame detection
✅ Generate processed video output
✅ Download detected video

---

## Analytics Dashboard

The application provides:

* Total detected objects
* Object class distribution
* Confidence analysis
* Detection summaries
* CSV export functionality

---

## Additional Features

The application also includes:

✅ Model information section
✅ Project description
✅ Detection statistics
✅ Download options

---

# 🗂️ Project Structure

```
Day15/
│
├── yolo_practice.py
│
├── object_detection.py
│
├── streamlit_app.py
│
├── requirements.txt
│
├── README.md
│
├── sample_input_images/
│
├── output_images/
│
├── dataset/              # Ignored by Git
│
├── yolov8n.pt            # Ignored by Git
│
└── runs/                 # Ignored by Git
```

---

# ⚙️ Installation & Usage

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run YOLO Practice

```bash
python yolo_practice.py
```

---

## Run Object Detection Pipeline

```bash
python object_detection.py
```

---

## Launch Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

---

# 🔐 GitHub Management

Large files are excluded using `.gitignore`.

Excluded files:

```
dataset/
yolov8n.pt
runs/
output_images/
```

This keeps the repository lightweight and deployment-friendly.

---

# 📚 Learning Outcomes

Through this project, I learned:

* Fundamentals of object detection
* Difference between classification and detection
* YOLOv8 architecture and inference workflow
* Bounding box prediction
* Confidence score interpretation
* Dataset handling for computer vision
* Building ML-powered Streamlit applications
* Deploying computer vision solutions

---

# 👨‍💻 Author

**Hadeed Jalani**

BS Computer Science
MLBench Summer Internship
