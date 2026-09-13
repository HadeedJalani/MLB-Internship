# 🚶 Smart People Counting System

> An intelligent computer vision application for detecting, tracking, and counting people in images and videos using **YOLOv8n, ByteTrack, OpenCV, and Streamlit**.

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?style=for-the-badge\&logo=python\&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-111111?style=for-the-badge)](https://github.com/ultralytics/ultralytics)
[![ByteTrack](https://img.shields.io/badge/Tracking-ByteTrack-00A67E?style=for-the-badge)](https://github.com/ifzhang/ByteTrack)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge\&logo=opencv\&logoColor=white)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20Application-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)](https://streamlit.io/)

---

## 📌 Overview

The **Smart People Counting System** is a computer vision application designed to detect and track people in images and video footage while generating meaningful occupancy and movement analytics.

The system combines the lightweight **YOLOv8n object detection model** with **ByteTrack multi-object tracking** to identify people, maintain tracking identities across consecutive frames, and calculate several types of people-counting metrics.

Beyond simply detecting people in individual frames, the project focuses on turning continuous video into structured information such as:

* 👥 Current people visible in a frame
* 📈 Peak occupancy during a video
* 🆔 Total unique tracking IDs observed
* 🚪 Entry and exit events using a counting line
* 🎯 People located inside a region of interest
* 📊 People-per-frame analytics
* 🎥 Annotated image and video outputs

The system is designed for practical use cases such as occupancy monitoring, public-space analysis, retail analytics, event monitoring, facility management, and intelligent surveillance research.

---

## 🎯 Project Objective

Traditional people counting often relies on manual observation or basic frame-by-frame detection.

However, a reliable counting system must answer more meaningful questions:

* How many people are visible right now?
* What was the highest occupancy during the video?
* How many different tracking identities appeared?
* How many people crossed a defined entrance or exit line?
* How many people are currently inside a selected area?
* Can people be tracked consistently across consecutive frames?

This project addresses these questions by combining:

```text
Detection → Tracking → Centroid Analysis → State Management → Analytics
```

The main objective is to build a modular and extensible system that converts raw visual footage into interpretable people-counting information.

---

## ✨ Key Features

### 🔍 Person Detection

The system uses the **COCO-pretrained YOLOv8n model** to detect people.

The model filters detections using the COCO class ID:

```python
classes=[0]
```

In the COCO dataset, class `0` represents the **person** category.

Each detection provides information such as:

* Bounding-box coordinates
* Object class
* Detection confidence
* Object location

---

### 🎯 Multi-Object Tracking

YOLOv8 identifies people in individual frames, while **ByteTrack** associates detections across consecutive frames.

This allows the system to assign persistent tracking IDs to people while they remain trackable.

Example:

```text
Frame 1 → Person ID 7
Frame 2 → Person ID 7
Frame 3 → Person ID 7
Frame 4 → Person ID 7
```

Tracking makes it possible to distinguish between:

* A person appearing repeatedly across multiple frames
* A genuinely new tracking identity
* A person moving through a monitored region
* A person crossing a counting line

---

### 👥 Current People Count

The current count represents the number of tracked people visible in the current frame.

Conceptually:

```python
current_count = len(people)
```

This value changes as people enter or leave the camera’s view.

---

### 📈 Peak Occupancy

Peak occupancy records the highest number of people visible at any point during the processed video.

```python
peak_count = max(peak_count, current_count)
```

This metric is useful for understanding the maximum observed crowd level in a scene.

---

### 🆔 Total Unique People Seen

The system maintains a set of observed tracking IDs:

```python
seen_ids.add(track_id)
```

At the end of processing, the number of observed tracking IDs is calculated as:

```python
total_unique_seen = len(seen_ids)
```

This represents the number of distinct tracking IDs observed during the processed clip.

> **Important:** This is not a guaranteed count of unique real-world humans. If the tracker loses a person and later assigns a new ID, the same individual may be counted more than once.

---

### 🚪 Counting-Line Analytics

The system supports an optional horizontal or vertical counting line.

For each tracked person, the system evaluates the position of the person’s centroid relative to the line.

A crossing is recorded only when the person changes from one side of the line to the other:

```text
Side A → Side B = Crossing
Side B → Side B = No crossing
Side A → Side A = No crossing
```

The default convention is:

```text
Negative side → Positive side = Entry
Positive side → Negative side = Exit
```

The previous side is stored separately for each tracking ID. This prevents a person who remains on the same side of the line for multiple frames from generating repeated crossing events.

---

### 🎯 Region-of-Interest Counting

The application supports an optional **Region of Interest**, or ROI.

The ROI is represented as an axis-aligned rectangle using normalized coordinates.

Example:

```python
ROI(
    x_min=0.10,
    x_max=0.90,
    y_min=0.10,
    y_max=0.90,
)
```

A person is considered inside the ROI when their centroid falls within the defined rectangle.

ROI counting can be useful for monitoring:

* Store entrances
* Waiting areas
* Exhibition zones
* Building lobbies
* Restricted spaces
* Event sections
* Production areas

---

## 🧠 System Architecture

The complete processing pipeline is structured as follows:

```text
                    ┌──────────────────────┐
                    │     Image / Video    │
                    └───────────┬──────────┘
                                │
                                ▼
                    ┌──────────────────────┐
                    │    Streamlit app.py  │
                    └───────────┬──────────┘
                                │
                                ▼
                    ┌──────────────────────┐
                    │  people_counter.py  │
                    └───────────┬──────────┘
                                │
             ┌──────────────────┼──────────────────┐
             ▼                  ▼                  ▼
     ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
     │    YOLOv8n    │  │   ByteTrack   │  │   Centroids   │
     │ Person Detect │  │ Track Persons │  │ Position Data │
     └───────┬───────┘  └───────┬───────┘  └───────┬───────┘
             │                  │                  │
             └──────────────────┼──────────────────┘
                                ▼
                    ┌──────────────────────┐
                    │   State Management   │
                    │ Peak / Unique / Line │
                    │ Entries / Exits / ROI│
                    └───────────┬──────────┘
                                │
                                ▼
                    ┌──────────────────────┐
                    │ Annotated Image / MP4│
                    └───────────┬──────────┘
                                │
                                ▼
                    ┌──────────────────────┐
                    │       outputs/       │
                    └──────────────────────┘
```

---

## 🛠️ Technology Stack

| Technology       | Purpose                             |
| ---------------- | ----------------------------------- |
| **Python 3.12**  | Core programming language           |
| **YOLOv8n**      | Person detection                    |
| **Ultralytics**  | YOLO model implementation           |
| **ByteTrack**    | Multi-object tracking               |
| **OpenCV**       | Image and video processing          |
| **NumPy**        | Numerical and coordinate processing |
| **Streamlit**    | Interactive web application         |
| **Pytest**       | Testing and validation              |
| **Git & GitHub** | Version control and project hosting |

---

## 📁 Project Structure

```text
Smart-People-Counting-System/
│
├── app.py
│   └── Streamlit application and user interface
│
├── people_counter.py
│   └── Person detection, tracking, counting, and analytics logic
│
├── requirements.txt
│   └── Python dependencies
│
├── README.md
│   └── Project documentation
│
├── coding_practice/
│   ├── 01_people_counting.py
│   │   └── Single-video people counting script
│   │
│   └── 02_batch_people_count.py
│       └── Batch processing script for multiple videos
│
├── sample_videos/
│   └── Sample people-counting videos
│
├── outputs/
│   └── Generated annotated images and videos
│
├── screenshots/
│   └── Application and output screenshots
│
└── .streamlit/
    └── config.toml
```

---

## 🔍 Person Detection

The system uses the COCO-pretrained YOLOv8n model and restricts detection to the person class.

The relevant tracking call follows this pattern:

```python
model.track(
    frame,
    persist=True,
    tracker="bytetrack.yaml",
    classes=[0],
)
```

The `classes=[0]` parameter ensures that the application focuses on people rather than detecting every object supported by the COCO dataset.

The `persist=True` parameter allows the tracking process to maintain identities across consecutive frames.

---

## 🎯 Tracking Identity Management

ByteTrack associates detections between frames and attempts to preserve a stable identity for each visible person.

A simplified example is:

```text
Frame 001 → Person ID 12
Frame 002 → Person ID 12
Frame 003 → Person ID 12
Frame 004 → Person ID 12
```

The system also maintains centroid information for tracked people.

A centroid is the approximate center point of a person’s bounding box:

```text
Bounding Box
     │
     ▼
Centroid
     │
     ├── Current count
     ├── Counting-line analysis
     └── ROI analysis
```

Centroid data provides a simple and useful representation of a person’s position within the frame.

---

## 📊 Counting Metrics

The application reports several related but distinct measurements.

### Current People

The number of tracked people visible in the current frame.

```python
current_count = len(people)
```

### Peak Occupancy

The maximum current count observed during the processed video.

```python
peak_count = max(peak_count, current_count)
```

### Total Unique Tracking IDs

The number of distinct tracking IDs observed throughout the processed clip.

```python
seen_ids.add(track_id)
```

```python
total_unique_seen = len(seen_ids)
```

These metrics should not be treated as interchangeable:

| Metric                  | Meaning                                              |
| ----------------------- | ---------------------------------------------------- |
| **Current people**      | People visible in the current frame                  |
| **Peak occupancy**      | Highest simultaneous count observed                  |
| **Unique tracking IDs** | Distinct IDs observed during processing              |
| **Entries**             | Detected negative-to-positive line crossings         |
| **Exits**               | Detected positive-to-negative line crossings         |
| **ROI count**           | People whose centroids are inside the configured ROI |

---

## 🚪 Counting-Line Logic

The counting line can be configured as either horizontal or vertical.

For each person, the system stores the previous side of the line according to that person’s tracking ID.

A crossing occurs only when the current side differs from the previously stored side.

```text
Previous Side = A
Current Side  = B
Result        = Crossing
```

```text
Previous Side = B
Current Side  = B
Result        = No crossing
```

```text
Previous Side = A
Current Side  = A
Result        = No crossing
```

The default event interpretation is:

```text
Negative → Positive = Entry
Positive → Negative = Exit
```

This per-track state management prevents repeated counting while a person remains on the same side of the line.

---

## 🎯 Region-of-Interest Logic

The optional ROI is defined using normalized coordinates.

Example:

```python
ROI(
    x_min=0.10,
    x_max=0.90,
    y_min=0.10,
    y_max=0.90,
)
```

The normalized coordinate system makes the ROI adaptable to different image and video resolutions.

A person is counted inside the ROI when the person’s centroid satisfies the following conditions:

```text
x_min ≤ centroid_x ≤ x_max
y_min ≤ centroid_y ≤ y_max
```

ROI counting is particularly useful when the application should monitor only a specific part of the scene.

---

## 🎥 Input and Output

### Supported Input Formats

The Streamlit application accepts:

* PNG
* JPG
* JPEG
* MP4
* AVI
* MOV
* MKV

### Generated Outputs

For videos, the application generates an annotated MP4 file.

Example:

```text
outputs/
└── mall_people_counted.mp4
```

For images, the application generates an annotated PNG file in the same output directory.

Annotated outputs may include:

* Person bounding boxes
* Tracking IDs
* Centroids
* Current count
* Counting line
* Entry and exit information
* ROI boundaries
* Additional monitoring information

---

## 🚀 Getting Started

### Prerequisites

Before running the project, install:

* Python 3.12 recommended
* pip
* Git
* A compatible environment for running YOLO inference

The application can run on a CPU, although a CUDA-enabled NVIDIA GPU may provide better performance for larger videos or higher-resolution inputs.

---

## 📥 Installation

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
```

Navigate to the project directory:

```bash
cd Smart-People-Counting-System
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activate the environment on macOS or Linux:

```bash
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The first YOLO model initialization may download the required `yolov8n.pt` weights automatically.

---

## ▶️ Running the Streamlit Application

Start the application with:

```powershell
streamlit run app.py
```

The Streamlit interface will open in the browser.

From the application, users can upload an image or video, configure the available counting options, process the input, and review the generated analytics and annotated output.

---

## 🖥️ Streamlit Usage

### For Images

The application produces an annotated PNG image containing the detected people and relevant visual information.

### For Videos

The application processes the video frame by frame and displays information such as:

* 👥 People currently visible
* 📈 Peak occupancy
* 🆔 Total unique tracked people seen
* 🚪 Counting-line crossings
* ➡️ Entries
* ⬅️ Exits
* 🎯 ROI count
* 📊 People-per-frame chart
* 🎥 Annotated MP4 output

The available options depend on the configuration implemented in the Streamlit interface.

---

## 💻 Coding Practice

### Process a Single Video

```powershell
python coding_practice/01_people_counting.py --video sample_videos/mall.mp4 --out outputs/mall_people_counted.mp4
```

### Use a Vertical Counting Line

```powershell
python coding_practice/01_people_counting.py --video sample_videos/mall.mp4 --out outputs/mall_people_counted.mp4 --line-orientation vertical --line-position 0.50
```

### Disable the Counting Line

```powershell
python coding_practice/01_people_counting.py --video sample_videos/mall.mp4 --out outputs/mall_people_counted.mp4 --no-line
```

### Enable the Example ROI

```powershell
python coding_practice/01_people_counting.py --video sample_videos/mall.mp4 --out outputs/mall_people_counted.mp4 --roi
```

### Process Multiple Videos in Batch Mode

```powershell
python coding_practice/02_batch_people_count.py --input-dir sample_videos --output-dir outputs
```

---

## 🧪 Testing and Validation

### Compile the Python Files

```powershell
python -m py_compile app.py
python -m py_compile people_counter.py
python -m py_compile coding_practice/01_people_counting.py
python -m py_compile coding_practice/02_batch_people_count.py
```

### Run the Application

```powershell
streamlit run app.py
```

### Recommended Validation Process

Test the system using at least five different people-counting videos.

Validation should include:

* Low-density scenes
* Crowded scenes
* People entering and leaving the frame
* People crossing the configured counting line
* People remaining on one side of the line
* People entering and leaving the ROI
* Different camera resolutions
* Different lighting conditions
* Partial occlusion
* Short and long video clips

---

## ⚙️ Deployment

The repository is structured for deployment through **Streamlit Community Cloud**.

The primary deployment files are:

```text
app.py
requirements.txt
people_counter.py
```

For hosted CPU environments, the application may use a frame-processing limit such as:

```python
HOSTED_FRAME_LIMIT = 450
```

This limit can help control processing time and resource usage in hosted environments.

For unrestricted local processing, the limit can be disabled:

```python
HOSTED_FRAME_LIMIT = None
```

With the limit disabled, the application can process frames until the video ends, subject to available hardware and system resources.

---

## ⚠️ Challenges and Limitations

### Occlusion

People may overlap with one another or become partially hidden. This can cause detections to disappear temporarily and make tracking more difficult.

### Tracking ID Switches

ByteTrack helps maintain consistent identities, but crowded scenes, long disappearances, and heavy occlusion can still cause a person to receive a different tracking ID.

### Camera Movement

Basic counting-line logic works best with a largely stationary camera.

When the camera moves, stationary people may appear to move relative to the image, which can produce inaccurate crossing events.

### Small or Distant People

YOLOv8n is optimized for efficiency, but very small, distant, blurred, or poorly illuminated people may be missed.

### Counting Semantics

Current count, peak occupancy, and unique tracking-ID count measure different aspects of the scene.

The unique tracking-ID count depends on the continuity and reliability of the tracking process.

### Environmental Conditions

Performance may be affected by:

* Poor lighting
* Motion blur
* Low-resolution footage
* Camera angle
* Dense crowds
* Reflections
* Partial visibility
* Unusual movement patterns

---

## 🔮 Future Improvements

Potential enhancements include:

* Advanced crowd-density estimation
* Improved tracking under heavy occlusion
* Camera-motion compensation
* Perspective-aware counting
* Multiple counting lines
* Multiple ROIs
* Direction-specific movement analysis
* Heatmap generation
* Real-time CCTV stream support
* Database-backed event storage
* Timestamped entry and exit logs
* CSV and JSON analytics export
* Automated occupancy alerts
* GPU-accelerated deployment
* Model comparison and benchmarking
* Advanced dashboard visualizations
* Privacy-preserving face blurring
* Multi-camera people tracking

---

## 🔐 Privacy Considerations

People-counting systems may process footage containing identifiable individuals.

When using real-world video, consider:

* Obtaining the necessary permissions
* Following applicable privacy regulations
* Avoiding unnecessary storage of raw footage
* Restricting access to processed outputs
* Using legally authorized datasets and recordings
* Applying anonymization techniques where appropriate

The system should be used responsibly and in accordance with applicable laws and organizational policies.

---

## 📚 Core Concepts Demonstrated

This project demonstrates practical implementation of:

* Computer Vision
* Deep Learning
* Object Detection
* YOLOv8
* Multi-Object Tracking
* ByteTrack
* Centroid-Based Analysis
* People Counting
* Occupancy Estimation
* Line-Crossing Detection
* Region-of-Interest Analysis
* Video Processing
* Streamlit Application Development
* Python Software Architecture
* Analytics Generation
* Automated Testing
* Modular System Design

---

## 🧩 Design Philosophy

The project separates the major responsibilities of the system into clear stages:

```text
Person Detection
        ↓
Multi-Object Tracking
        ↓
Centroid Calculation
        ↓
Counting and Rule Evaluation
        ↓
State Management
        ↓
Analytics
        ↓
Annotated Output
```

This modular structure improves:

* Maintainability
* Debugging
* Reusability
* Testing
* Scalability
* Future model integration

The system can therefore be extended without requiring the entire application to be redesigned.

---

## 📦 Dataset and Sample Videos

For meaningful evaluation, use at least five legally authorized people-counting videos.

Possible sources include:

* Pexels
* Pixabay
* Your own recordings
* Other datasets with suitable redistribution licenses

Before using any video, verify its licensing terms.

Raw video files should not be committed to GitHub unless redistribution is explicitly permitted.

---

## 🤝 Contributing

Contributions, improvements, and technical suggestions are welcome.

To create a feature branch:

```bash
git checkout -b feature/your-feature
```

After making changes:

```bash
git add .
git commit -m "Add your feature"
git push origin feature/your-feature
```

When contributing, please:

* Keep changes focused
* Preserve the modular project structure
* Add tests where appropriate
* Avoid committing unnecessary generated files
* Document new functionality
* Respect dataset and video licenses

---

## 📄 License

This project is provided for educational, research, and development purposes.

Before using the system in a production environment, review the licensing requirements of the software libraries, model weights, datasets, and video footage involved.

---

## 👨‍💻 Author

### Hadeed Jalani

AI and Machine Learning Developer focused on building practical solutions in:

```text
Python
Artificial Intelligence
Machine Learning
Deep Learning
Computer Vision
Object Detection
Object Tracking
Data Analytics
```

---

## ⭐ Acknowledgements

This project is built upon the work of the open-source machine learning and computer vision community, including:

* Ultralytics YOLO
* ByteTrack
* OpenCV
* Streamlit
* NumPy
* Python

These technologies provide the foundation for building modern intelligent video-analysis applications.

---

<p align="center">
  <strong>🚶 Smart People Counting System</strong>
  <br>
  Detection • Tracking • Occupancy • Movement Analytics
  <br><br>
  <strong>Built with Python, YOLOv8n, ByteTrack, OpenCV & Streamlit</strong>
</p>
