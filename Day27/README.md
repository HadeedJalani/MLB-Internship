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
```
The bounding box represents the location of the detected object inside the image.

🔍 Image Classification vs Object Detection vs Image Segmentation
Task	Output
Image Classification	Predicts one label for the entire image
Object Detection	Detects multiple objects and their locations
Image Segmentation	Classifies individual pixels
Example

Consider an image containing a person and a car.

Image Classification

Scene: Street

Object Detection

Person → Bounding Box
Car → Bounding Box

Image Segmentation

Every pixel belonging to Person → Mask
Every pixel belonging to Car → Mask
🤖 What is YOLO?

YOLO stands for:

You Only Look Once

YOLO is a popular family of real-time object detection models.

Unlike traditional detection pipelines that perform multiple stages separately, YOLO processes the entire image in a single forward pass through a neural network.

This makes YOLO:

⚡ Fast
🎯 Accurate
🚀 Suitable for real-time applications

YOLO is widely used in:

Autonomous Vehicles
Security Systems
Smart Surveillance
Robotics
Retail Analytics
Sports Analysis
Traffic Monitoring
Industrial Automation
🧠 YOLO Detection Pipeline

The detection pipeline used in this project is:

Input Image / Video
        │
        ▼
Image Preprocessing
        │
        ▼
YOLO11 Model
        │
        ▼
Object Predictions
        │
        ├── Class Labels
        ├── Bounding Boxes
        └── Confidence Scores
        │
        ▼
Confidence Filtering
        │
        ▼
Annotated Output
🏗️ Project Structure
Day27/
│
├── sample_inputs/
│   │
│   ├── images/
│   │   ├── sample_image_1.jpg
│   │   ├── sample_image_2.jpg
│   │   └── ...
│   │
│   └── videos/
│       ├── sample_video_1.mp4
│       └── sample_video_2.mp4
│
├── outputs/
│   │
│   ├── images/
│   │   └── detected_images/
│   │
│   └── videos/
│       └── detected_videos/
│
├── test_detection.py
│
├── test_video_detection.py
│
├── app.py
│
├── requirements.txt
│
├── runtime.txt
│
├── .gitignore
│
└── README.md
📦 Technologies Used
Technology	Purpose
Python	Core Programming Language
Ultralytics	YOLO11 Implementation
YOLO11	Object Detection Model
OpenCV	Image and Video Processing
Streamlit	Interactive Web Application
NumPy	Numerical Processing
Pillow	Image Handling
🤖 YOLO Model Used

This project uses:

YOLO11 Nano (yolo11n.pt)

YOLO11n was selected because it provides an excellent balance between:

Inference Speed
Detection Accuracy
Low Computational Requirements
Deployment Compatibility

The model is pre-trained on the Microsoft COCO Dataset.

🏷️ Detectable Object Classes

The COCO dataset contains 80 common object categories.

Examples include:

People & Animals
Person
Dog
Cat
Horse
Bird
Cow
Sheep
Elephant
Vehicles
Car
Bus
Truck
Motorcycle
Bicycle
Train
Airplane
Everyday Objects
Chair
Couch
Bed
Dining Table
Laptop
Keyboard
Mouse
TV
Sports Objects
Sports Ball
Tennis Racket
Baseball Bat
Skis
Surfboard
📊 Object Detection Results

The YOLO11 model was tested on multiple images containing different objects.

Example detection output:

Objects detected: 4

1. Cat       → 92.79%
2. Cat       → 91.80%
3. Remote    → 66.59%
4. Couch     → 49.89%

Another example:

Objects detected: 5

1. Bus       → 94.02%
2. Person    → 88.82%
3. Person    → 87.83%
4. Person    → 85.58%
5. Person    → 62.19%

The model successfully detected multiple objects simultaneously and generated bounding boxes with confidence scores.

🎨 Bounding Box Visualization

Each detected object is visualized using:

Colored Bounding Boxes
Class Labels
Confidence Scores

Example:

┌─────────────────────────┐
│ Person 92%             │
│                         │
│                         │
│        PERSON           │
│                         │
└─────────────────────────┘

Different object classes are assigned different colors to improve visualization and make detection results easier to interpret.

🖼️ Image Detection Workflow
Upload Image
      │
      ▼
Load Image with OpenCV
      │
      ▼
YOLO11 Inference
      │
      ▼
Filter Predictions
      │
      ├── Confidence Threshold
      │
      ▼
Draw Bounding Boxes
      │
      ├── Class Name
      └── Confidence Score
      │
      ▼
Display Results
      │
      ▼
Download Processed Image
🎥 Video Detection Workflow
Upload Video
      │
      ▼
Extract Frames
      │
      ▼
Run YOLO Detection
      │
      ▼
Draw Object Annotations
      │
      ▼
Process Next Frame
      │
      ▼
Generate Output Video
      │
      ▼
Display Input + Output
      │
      ▼
Download Processed Video
⚡ Confidence Threshold

The application allows users to control the minimum confidence required for an object to be displayed.

For example:

Confidence Threshold = 0.50

This means:

Confidence ≥ 50% → Display Detection
Confidence < 50% → Ignore Detection

Increasing the threshold:

Reduces false positives
Displays only stronger predictions

Decreasing the threshold:

Detects more objects
May include weaker predictions
🧪 Image Testing

The YOLO model was tested on 10 different images.

The test dataset included:

Cats
Couch
Bus
People
Sports Scenes
Bedroom Scenes
Vehicles
Traffic Scenes
Teddy Bears
Construction / Street Objects

The testing script automatically:

Loads all images
Runs YOLO inference
Extracts object predictions
Prints detection results
Saves annotated images

Example command:

python test_detection.py
🎥 Video Testing

The project also supports object detection on videos.

The video testing pipeline:

Loads the input video
Reads frames sequentially
Runs YOLO inference on every frame
Draws bounding boxes
Saves the processed video

Run:

python test_video_detection.py

Processed videos are saved inside:

outputs/videos/
🖥️ Streamlit Application

The project includes an interactive Streamlit application.

Users can:

Step 1

Select input type:

Image
or
Video
Step 2

Upload media.

Step 3

Adjust the confidence threshold.

Step 4

Run YOLO object detection.

Step 5

Compare the input and output side-by-side.

┌──────────────────────┬──────────────────────┐
│                      │                      │
│    INPUT MEDIA       │    DETECTED OUTPUT   │
│                      │                      │
└──────────────────────┴──────────────────────┘
Step 6

Download the processed output.

🚀 Installation
1️⃣ Clone the Repository
git clone https://github.com/HadeedJalani/MLB-Internship.git

Navigate to the project folder:

cd MLB-Internship/Day27
2️⃣ Create Virtual Environment

Windows:

python -m venv venv

Activate:

venv\Scripts\activate

Linux / Mac:

source venv/bin/activate
3️⃣ Install Dependencies
pip install -r requirements.txt
▶️ Running the Application

Launch the Streamlit application:

streamlit run app.py

Streamlit will provide a local URL similar to:

http://localhost:8501

Open the URL in your browser.

📋 Requirements

The main dependencies include:

streamlit
ultralytics
opencv-python-headless
numpy
pillow

Install them using:

pip install -r requirements.txt
☁️ Deployment

The application is designed to be deployment-friendly and supports deployment on:

Streamlit Community Cloud
Local Streamlit Server
Cloud-based Python environments

The application automatically downloads the YOLO11 pre-trained model when required.

🔗 Deployment
🌐 Live Streamlit Application

Add your deployed Streamlit URL here

[Your Streamlit App URL]
💻 GitHub Repository

https://github.com/HadeedJalani/MLB-Internship/tree/main/Day27

🌍 Real-World Applications

Object Detection is used across many industries.

🚗 Autonomous Vehicles

Detect:

Cars
Pedestrians
Traffic Signs
Bicycles
🏪 Retail Analytics

Detect:

Products
Customers
Empty Shelves
🛡️ Security Systems

Detect:

People
Vehicles
Suspicious Objects
🏭 Industrial Automation

Detect:

Manufacturing Defects
Equipment
Products
🏥 Healthcare

Detect:

Medical Instruments
Abnormalities
Biological Objects
🤖 Robotics

Robots use object detection to understand their surroundings and interact with objects.

⚠️ Challenges Faced
🎥 Video Rendering in Streamlit

Initially, uploaded videos were successfully processed but were not displaying correctly in the Streamlit interface.

Solution

The application was updated to:

Save uploaded videos temporarily
Process videos frame-by-frame
Generate a compatible output video
Display input and output videos side-by-side
🧠 Model Loading Time

YOLO models require loading weights before inference.

Solution

The YOLO model was cached using Streamlit resource caching to avoid repeatedly loading the model.

⚡ Video Processing Performance

Running inference on every video frame can be computationally expensive.

Solution

The YOLO11 Nano model was selected because it provides fast inference while maintaining strong detection performance.

🎨 Bounding Box Visibility

Default visualization can sometimes make labels difficult to read.

Solution

The application ensures:

Clearly visible bounding boxes
High contrast labels
Confidence score display
Different colors for different object classes
📈 Key Learnings

Through this project, the following concepts were explored:

Object Detection Fundamentals
Bounding Boxes
Class Labels
Confidence Scores
YOLO Architecture
Pre-trained Model Inference
Image Detection
Video Detection
Frame-by-frame Processing
OpenCV Visualization
Streamlit Deployment
Media File Handling
🔮 Future Improvements

Potential improvements include:

📹 Real-time webcam detection
🎯 Object tracking across video frames
🧠 Custom YOLO model training
📊 Detection analytics dashboard
🔢 Object counting
🗺️ Object location tracking
⚡ GPU inference support
🎥 Live RTSP camera support
📈 Detection history and analytics
🤖 Multi-model comparison
🎓 Internship Context

This project was developed as part of the:

MLBench Summer Internship
Day 27 — Introduction to Object Detection

The objective was to understand the complete object detection workflow using a pre-trained YOLO model, from loading a model and running inference to building and deploying an interactive application.

📸 Expected Output

The application produces results similar to:

Input Image / Video
        ↓
YOLO11 Object Detection
        ↓
Bounding Boxes
        ↓
Class Labels
        ↓
Confidence Scores
        ↓
Annotated Output
👨‍💻 Author
<div align="center">
Hadeed Jalani

Final-Year BSCS Student | AI/ML & Full-Stack Developer

