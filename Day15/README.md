# Day 15 - YOLOv8 Object Detection System 🚀

## 📌 Overview

This project implements an end-to-end **Object Detection System using YOLOv8** as part of the MLBench Internship.

Unlike traditional image classification, where a model predicts a single class for an entire image, **object detection identifies multiple objects inside an image and locates them using bounding boxes**.

In this project, a pretrained **YOLOv8 Nano model (`yolov8n.pt`)** is used to detect objects from images and videos, analyze detection results, generate visual outputs, and deploy the complete solution using Streamlit.

---

# 🎯 Project Objectives

The main objectives of this task were:

* Understand the fundamentals of Object Detection
* Learn YOLOv8 architecture and workflow
* Perform inference using a pretrained YOLO model
* Detect objects from images and videos
* Generate bounding box visualizations
* Analyze detection results
* Build an interactive computer vision application using Streamlit

---

# 🧠 Object Detection vs Image Classification

| Image Classification                    | Object Detection                                  |
| --------------------------------------- | ------------------------------------------------- |
| Predicts one label for an entire image  | Detects multiple objects in an image              |
| Does not identify object location       | Provides object location using bounding boxes     |
| Example: "This image contains a helmet" | Example: "There are 3 helmets at these locations" |

---

# 🤖 YOLOv8

YOLO (**You Only Look Once**) is a real-time object detection algorithm that predicts:

* Object classes
* Bounding box coordinates
* Confidence scores

This project uses:

**Model:** YOLOv8 Nano (`yolov8n.pt`)

Reasons for using YOLOv8 Nano:

* Lightweight architecture
* Fast inference speed
* Suitable for real-time applications
* Good balance between speed and accuracy

---

# 📂 Project Structure

```
Day15/
│
├── yolo_practice.py              # YOLOv8 basic practice implementation
│
├── object_detection.py           # Complete object detection pipeline
│
├── streamlit_app.py              # Interactive detection dashboard
│
├── requirements.txt              # Required dependencies
│
├── README.md                     # Project documentation
│
├── sample_input_images/          # Sample images for testing
│
├── output_images/                # Generated detection results
│
├── dataset/                      # Helmet Detection dataset (ignored)
│
├── yolov8n.pt                    # YOLO pretrained weights (ignored)
│
└── runs/                         # YOLO training/inference outputs (ignored)
```

---

# 📊 Dataset Used

## Helmet Detection Dataset

Dataset Source:

Roboflow Public Dataset

Format:

* YOLO format
* Image + annotation files

The dataset contains images used for object detection and evaluation.

The dataset is excluded from GitHub because of its large size.

---

# 🛠️ Technologies Used

* Python
* YOLOv8
* Ultralytics
* OpenCV
* PyTorch
* Pandas
* Streamlit

---

# 🔍 YOLO Practice Implementation

File:

```
yolo_practice.py
```

This script demonstrates basic YOLOv8 functionality:

### Features:

✅ Load pretrained YOLOv8 model
✅ Perform single image detection
✅ Perform multiple image detection
✅ Display confidence scores
✅ Extract bounding boxes
✅ Display detected class labels
✅ Save prediction results

---

# 🚀 Object Detection Pipeline

File:

```
object_detection.py
```

The complete detection workflow includes:

### 1. Dataset Loading

Loads images from the helmet detection dataset.

### 2. Model Inference

The pretrained YOLOv8 model processes images and predicts:

* Detected objects
* Bounding boxes
* Confidence scores

### 3. Visualization

The output contains:

* Bounding boxes
* Class labels
* Confidence values

### 4. Result Generation

Detection results are saved for further analysis.

---

# 📈 Detection Analysis

The system generates:

* Total number of detections
* Detected object classes
* Confidence scores
* Detection summaries

These results help evaluate model performance and understand prediction behavior.

---

# 🌐 Streamlit Application

File:

```
streamlit_app.py
```

The project includes an interactive computer vision dashboard.

## Features:

### Image Detection

Users can:

* Upload an image
* Run YOLO detection
* View bounding boxes
* Download processed output

---

### Video Detection

Users can:

* Upload videos
* Perform frame-by-frame detection
* Generate processed videos
* Download results

---

### Analytics Dashboard

The application provides:

* Detection statistics
* Object counts
* Confidence analysis
* Visual summaries

---

### Additional Features

The application also includes:

✅ Model information
✅ About section
✅ CSV export of detections
✅ Download options

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/HadeedJalani/MLB-Internship.git
```

Navigate to Day15:

```bash
cd MLB-Internship/Day15
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

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

# 📌 Sample Results

The model generates output images containing:

* Bounding boxes
* Class labels
* Confidence scores

Example workflow:

Input Image → YOLOv8 Model → Detection → Annotated Output

---

# 🔐 GitHub File Management

Large files are excluded using `.gitignore`.

Excluded:

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
* YOLOv8 model usage
* Bounding box prediction
* Confidence score interpretation
* Dataset handling for computer vision
* Building production-style Streamlit ML applications

---

# 👨‍💻 Author

**Hadeed Jalani**

BS Computer Science
MLBench Internship

---

# ⭐ Future Improvements

Possible improvements:

* Train YOLOv8 on custom helmet dataset
* Improve detection accuracy
* Add real-time webcam detection
* Deploy using cloud GPU infrastructure
* Add model performance metrics
