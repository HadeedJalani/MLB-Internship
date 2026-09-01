<h1 align="center">🚦 Smart Vehicle Counting System</h1>

<h3 align="center">Multi-Vehicle Detection, Tracking & Line-Crossing Analytics with YOLO11 + ByteTrack / BoT-SORT</h3>

<p align="center">
  <strong>Upload a traffic video, track vehicles with persistent IDs, detect line crossings, and download the annotated counting video.</strong>
</p>

<p align="center">
  Built using <strong>Ultralytics YOLO11</strong> for object detection,
  <strong>ByteTrack</strong> and <strong>BoT-SORT</strong> for multi-object tracking,
  <strong>OpenCV</strong> for video processing and line-crossing logic,
  and <strong>Streamlit</strong> for the interactive web application.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/YOLO-11-00FFFF?style=for-the-badge&logo=ultralytics&logoColor=black" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Model-YOLO11n-orange" />
  <img src="https://img.shields.io/badge/Trackers-ByteTrack%20%7C%20BoT--SORT-blue" />
  <img src="https://img.shields.io/badge/Deployment-Streamlit-red" />
</p>

---

## 📌 Project Overview

Traffic monitoring systems require more than simple object detection. They need to identify and count **unique vehicles** moving through a defined region without repeatedly counting the same vehicle across multiple frames.

The **Smart Vehicle Counting System** is a computer vision application that processes traffic surveillance videos to:

* Detect vehicles frame-by-frame using YOLO11
* Track individual vehicles using persistent tracking IDs
* Detect when vehicles cross a virtual counting line
* Prevent duplicate counting of the same vehicle
* Determine traffic movement direction
* Generate class-wise vehicle statistics
* Produce an annotated output video with real-time analytics
* Provide an interactive Streamlit interface for video processing

### Complete Processing Pipeline

```text
Input Traffic Video
        │
        ▼
YOLO11 Detection (Per Frame)
        │
        ▼
Tracker (ByteTrack / BoT-SORT)
        │
        ├── Persistent Track IDs
        ├── Vehicle Class Filtering
        └── Centroid Trajectory Calculation
        │
        ▼
Virtual Line Crossing Detection
(y_previous vs. y_current)
        │
        ├── Unique Track ID Check
        │   └── Prevents Duplicate Counting
        │
        └── Direction Classification
            ├── Upward
            └── Downward
        │
        ▼
Annotated Frame
(Boxes + IDs + Centroids + Line + HUD)
        │
        ▼
Output Video + Analytics Dashboard
```

---

## 🎯 Objective

The primary objective of this project is to build an intelligent traffic surveillance system capable of answering:

> **"How many total and unique vehicles passed through this road section, what types of vehicles were they, and in which direction were they moving?"**

The system combines **object detection, multi-object tracking, coordinate-based movement analysis, and line-crossing logic** to provide reliable traffic-flow analytics.

---

## 🚀 Key Features

| Feature                           | Description                                                                                           |
| --------------------------------- | ----------------------------------------------------------------------------------------------------- |
| 🧠 **YOLO11 Vehicle Detection**   | Detects and classifies target vehicle classes: `car`, `bus`, `truck`, and `motorcycle`.               |
| 🔗 **Selectable Tracker**         | Choose between ByteTrack for speed or BoT-SORT for stronger identity tracking and occlusion handling. |
| 🆔 **Persistent Tracking IDs**    | Assigns unique IDs to individual vehicles and maintains their identity across frames.                 |
| 📏 **Virtual Counting Line**      | Uses a configurable horizontal region-of-interest line for vehicle crossing detection.                |
| 🚫 **Duplicate Prevention**       | Counts each vehicle only once using its unique tracking ID.                                           |
| ↕️ **Directional Counting**       | Separately counts vehicles moving upward and downward.                                                |
| 🎨 **Per-ID Bounding Box Colors** | Generates deterministic colors so individual vehicles remain visually distinguishable.                |
| 📊 **Live Analytics Dashboard**   | Displays vehicle totals, class-wise counts, unique IDs, and directional statistics.                   |
| 💾 **Downloadable Results**       | Allows users to download the fully annotated processed MP4 video.                                     |
| 🌐 **Streamlit Interface**        | Provides an interactive browser-based interface for uploading and analyzing traffic videos.           |

---

## ⚡ How Vehicle Counting & Duplicate Prevention Work

A simple object detection system would detect the same vehicle in every frame. If a car appears in 500 frames, naïvely counting every detection could result in the same car being counted hundreds of times.

This project solves the problem by combining **multi-object tracking** with **persistent track IDs**.

### 1. Centroid Tracking

For every detected vehicle, the center point of its bounding box is calculated:

```text
x_center = (x1 + x2) / 2
y_center = (y1 + y2) / 2
```

The vertical centroid position is then monitored across consecutive frames.

### 2. Line Crossing Detection

A virtual horizontal line is positioned across the traffic road.

The system compares the previous and current centroid positions against the line:

**Downward Crossing**

```text
y_previous < Y_line
AND
y_current >= Y_line
```

**Upward Crossing**

```text
y_previous > Y_line
AND
y_current <= Y_line
```

When one of these conditions becomes true, a crossing event is registered.

### 3. Persistent Track ID Filtering

Each vehicle receives a persistent `track_id` from the selected tracker.

Once a vehicle crosses the counting line, its ID is stored inside a `counted_ids` set:

```python
counted_ids = set()
```

Before registering a new count, the system checks whether the ID has already been counted.

This ensures that:

```text
Vehicle detected
       ↓
Vehicle tracked
       ↓
Vehicle crosses line
       ↓
Track ID checked
       ↓
Already counted?
   ↙          ↘
 YES           NO
  ↓             ↓
Ignore      Register Count
              ↓
        Save Track ID
```

As a result, the same vehicle is counted **only once**, even if it remains visible for many frames.

---

## 🔄 Tracker Comparison

The application supports two multi-object tracking algorithms.

| Tracker       | Strengths                                                                                   | Trade-off                                                |
| ------------- | ------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| **ByteTrack** | Fast, lightweight, and efficient for clear traffic footage.                                 | Can be weaker when vehicles heavily occlude one another. |
| **BoT-SORT**  | Combines motion and appearance-based re-identification and performs better under occlusion. | Higher computational cost compared with ByteTrack.       |

### When to Use Each Tracker

**ByteTrack**

Best suited for:

* Clear traffic footage
* Low-to-moderate vehicle density
* Faster processing
* Resource-constrained environments

**BoT-SORT**

Best suited for:

* Crowded traffic
* Frequent vehicle overlap
* More challenging tracking environments
* Situations where maintaining vehicle identity is important

---

## 🚗 Target Vehicle Classes

The system focuses on four vehicle classes from the COCO dataset.

| Icon | Vehicle Class  | COCO Index | Target Application                                 |
| :--: | -------------- | :--------: | -------------------------------------------------- |
|  🚗  | **Car**        |     `2`    | Sedans, hatchbacks, SUVs, and other passenger cars |
|  🏍️ | **Motorcycle** |     `3`    | Motorcycles, scooters, and other two-wheelers      |
|  🚌  | **Bus**        |     `5`    | Passenger and city transit buses                   |
|  🚛  | **Truck**      |     `7`    | Commercial cargo and delivery trucks               |

---

## 🤖 Model Used

This project uses the pretrained **YOLO11n (Nano)** model from Ultralytics.

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
```

YOLO11n was selected because it provides a good balance between:

* Detection performance
* Inference speed
* Computational requirements
* Streamlit deployment compatibility

### 💡 Tuning Tip

If distant or low-resolution vehicles are not being detected reliably, consider adjusting the confidence threshold.

For example:

```text
0.35 – 0.50
```

Alternatively, the model can be upgraded from:

```text
yolo11n.pt
```

to:

```text
yolo11s.pt
```

for potentially improved detection accuracy at the cost of additional computational requirements.

---

## 📂 Project Structure

```text
Day-30/
│
├── counter/
│   ├── __init__.py
│   └── vehicle_counter.py
│       # Core VehicleCounter class
│       # Detection + Tracking + Line Crossing
│
├── app.py
│   # Interactive Streamlit frontend
│
├── requirements.txt
│   # Python dependencies
│
└── README.md
    # Project documentation
```

---

## ⚙️ System Architecture

The project is divided into two primary components:

### `counter/vehicle_counter.py`

This module contains the core computer vision and vehicle counting logic.

Responsibilities include:

* Initializing the YOLO11 model
* Configuring the selected tracking algorithm
* Processing video frames
* Detecting target vehicle classes
* Maintaining persistent tracking IDs
* Calculating vehicle centroids
* Detecting virtual line crossings
* Determining movement direction
* Preventing duplicate vehicle counts
* Generating annotated frames
* Maintaining class-wise and direction-wise statistics

The tracker is configured using Ultralytics tracking configurations:

```text
bytetrack.yaml
botsort.yaml
```

Persistent tracking is performed using:

```python
model.track(..., persist=True)
```

---

### `app.py`

The Streamlit application acts as the frontend layer of the project.

It is responsible for:

* Rendering the web interface
* Accepting uploaded traffic videos
* Providing tracker selection
* Providing confidence threshold controls
* Managing temporary uploaded files
* Executing video processing
* Displaying processing progress
* Showing input and output videos
* Displaying analytics
* Providing the processed video for download

---

## 🔬 Processing Workflow

The complete processing workflow can be summarized as follows:

```text
1. Upload Traffic Video
          ↓
2. Select Tracker
   ByteTrack / BoT-SORT
          ↓
3. Set Confidence Threshold
          ↓
4. Initialize YOLO11
          ↓
5. Read Video Frame
          ↓
6. Detect Vehicles
          ↓
7. Assign Persistent Track IDs
          ↓
8. Filter Vehicle Classes
          ↓
9. Calculate Vehicle Centroids
          ↓
10. Compare Centroid Movement
          ↓
11. Check Virtual Line Crossing
          ↓
12. Determine Direction
          ↓
13. Check counted_ids
          ↓
14. Update Statistics
          ↓
15. Draw Bounding Boxes + IDs + HUD
          ↓
16. Write Annotated Frame
          ↓
17. Generate Output Video
          ↓
18. Display Analytics + Download
```

---

## 🌐 Streamlit Application

The project includes an interactive Streamlit web application that allows users to process traffic videos directly from their browser.

### Application Features

#### 📤 Video Upload

Supports common video formats including:

* MP4
* AVI
* MOV
* MKV

#### ⚙️ Detection Settings

Users can configure:

* Detection confidence threshold
* Tracking algorithm

  * ByteTrack
  * BoT-SORT

#### 📊 Processing Progress

A live progress bar provides feedback while the system processes the uploaded video frame-by-frame.

#### 🎬 Video Playback

The interface presents the:

* Original input video
* Processed annotated video

This allows users to visually compare the original footage with the detection and counting results.

#### 📈 Counting Metrics

The dashboard provides statistics including:

* Total counted vehicles
* Unique tracking IDs
* Total processed frames
* Processing runtime
* Class-wise vehicle counts

#### 🚗 Vehicle Breakdown

Vehicle counts are displayed separately for:

* Cars
* Motorcycles
* Buses
* Trucks

#### ↕️ Directional Statistics

The application tracks traffic movement in two directions:

* ⬆️ Upward
* ⬇️ Downward

#### ⬇️ Video Export

Once processing is complete, users can download the generated annotated MP4 video directly from the application.

---

## 📊 Analytics Generated

The system maintains multiple layers of traffic statistics.

### Total Vehicle Count

Represents the number of unique vehicles that successfully crossed the counting line.

### Unique Tracking IDs

Represents the number of unique vehicle identities observed by the tracker.

### Class-wise Counts

Vehicles are categorized into:

```text
Car
Motorcycle
Bus
Truck
```

### Direction-wise Counts

Vehicles are categorized according to their movement:

```text
Upward
Downward
```

These statistics provide a more useful representation of traffic flow than raw frame-level detections.

---

## 💾 Output Video

The generated output video contains a visual analytics overlay including:

* Vehicle bounding boxes
* Persistent tracking IDs
* Vehicle class labels
* Vehicle centroids
* Virtual counting line
* Direction information
* Live vehicle counts
* Traffic statistics HUD

The processed video can be downloaded directly from the Streamlit application.

---

## 🚀 Running Locally

### 1. Clone the Repository

```bash
git clone https://github.com/HadeedJalani/MLB-Internship.git
```

Navigate to the Day-30 project:

```bash
cd MLB-Internship/Day-30
```

### 2. Create a Virtual Environment

It is recommended to use a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit Application

```bash
streamlit run app.py
```

The application will typically be available at:

```text
http://localhost:8501
```

### 5. Model Download

The `yolo11n.pt` model weights are automatically downloaded by Ultralytics when the model is initialized for the first time.

---

## 📦 Dependencies

The project relies on the following primary libraries:

```text
ultralytics
opencv-python
streamlit
numpy
```

The exact dependency versions are maintained in:

```text
requirements.txt
```

---

## 🛠️ Technology Stack

| Technology             | Purpose                                                                |
| ---------------------- | ---------------------------------------------------------------------- |
| **Python 3.11**        | Primary programming language                                           |
| **Ultralytics YOLO11** | Real-time vehicle detection and tracking integration                   |
| **ByteTrack**          | Fast multi-object persistent tracking                                  |
| **BoT-SORT**           | Appearance and motion-based multi-object tracking                      |
| **OpenCV**             | Video processing, frame manipulation, drawing, and line-crossing logic |
| **Streamlit**          | Interactive web application and analytics dashboard                    |
| **NumPy**              | Numerical and coordinate-based calculations                            |

---

## 🎓 Key Learnings

This project provided hands-on experience with several important computer vision and software engineering concepts.

### Computer Vision

* Real-time object detection using YOLO11
* Processing video streams frame-by-frame
* Bounding box coordinate manipulation
* Centroid calculation
* Region-of-interest and virtual line logic
* Video annotation and rendering

### Multi-Object Tracking

* Persistent object identities
* Track ID management
* ByteTrack integration
* BoT-SORT integration
* Handling vehicle movement across multiple frames

### Traffic Analytics

* Virtual line-crossing detection
* Direction-aware vehicle counting
* Class-wise traffic statistics
* Duplicate counting prevention
* Unique vehicle identification

### Streamlit Development

* Building interactive computer vision applications
* File upload handling
* Temporary file management
* Session-based application execution
* Real-time progress feedback
* Video playback
* Downloadable output generation

### Deployment

* Preparing computer vision applications for web deployment
* Managing Python dependencies
* Handling model downloads
* Optimizing lightweight YOLO inference for cloud environments

---

## 🔮 Possible Future Improvements

The current system provides a solid foundation for intelligent traffic analytics. Future improvements could include:

* 🚦 Multiple virtual counting lines
* 🛣️ Lane-wise vehicle counting
* 📍 Configurable line position directly from the UI
* 📈 Historical traffic analytics
* 📊 Traffic volume graphs over time
* 🕒 Vehicle speed estimation
* 🚨 Wrong-way vehicle detection
* 🔴 Red-light violation detection
* 🗺️ Multi-camera traffic monitoring
* ☁️ Cloud-based analytics storage
* 🤖 Larger YOLO models for improved detection accuracy
* 📱 Mobile-responsive monitoring dashboard

---

## 🌐 Live Demo & Submission

### 🚀 Live Streamlit Application

[Open Smart Vehicle Counting System](https://mlb-internship-day30.streamlit.app/)

### 💻 GitHub Repository

[MLB Internship Repository](https://github.com/HadeedJalani/MLB-Internship)

---

## 📝 Project Summary

The **Smart Vehicle Counting System** demonstrates how modern computer vision techniques can be combined to create a practical traffic monitoring solution.

Instead of simply detecting vehicles in individual frames, the system combines:

```text
YOLO11
   +
ByteTrack / BoT-SORT
   +
Centroid Tracking
   +
Virtual Line Crossing
   +
Persistent Track IDs
   +
Direction Classification
   +
OpenCV Video Processing
   +
Streamlit Dashboard
```

This enables the application to move from basic **object detection** toward meaningful **traffic-flow analytics**, while minimizing duplicate vehicle counts and providing an interactive interface for real-world use.

---

## 👨‍💻 Author

### **Hadeed Jalani**

**Computer Science | Artificial Intelligence | Machine Learning | Computer Vision | Full-Stack Development**

Focused on building intelligent software solutions by combining **AI/ML, computer vision, and modern full-stack technologies**.

---

<p align="center">
  <strong>🚦 Smart Traffic Analytics • YOLO11 • Multi-Object Tracking • Computer Vision</strong>
</p>

<p align="center">
  Made with Python, OpenCV, Ultralytics YOLO11 & Streamlit
</p>
