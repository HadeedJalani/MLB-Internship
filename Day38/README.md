# 🛡️ Intelligent Security Monitoring + Image Segmentation

> An intelligent computer vision system combining **YOLOv8n, ByteTrack, polygon-based security zones, event detection, CSV logging, video analytics, and classical image segmentation** through an interactive Streamlit application.

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?style=for-the-badge\&logo=python\&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-111111?style=for-the-badge)](https://github.com/ultralytics/ultralytics)
[![ByteTrack](https://img.shields.io/badge/Tracking-ByteTrack-00A67E?style=for-the-badge)](https://github.com/ifzhang/ByteTrack)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge\&logo=opencv\&logoColor=white)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20Application-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)](https://streamlit.io/)
[![Ngrok](https://img.shields.io/badge/Deployment-ngrok-1F1F1F?style=for-the-badge)](https://ngrok.com/)

---

## 📌 Overview

The **Intelligent Security Monitoring + Image Segmentation** project combines two complementary computer vision workflows into one modular application.

The first component focuses on **intelligent security monitoring**, where people are detected and tracked through video footage while their movement through configurable polygonal regions is analyzed.

The second component focuses on **classical image segmentation**, providing multiple thresholding techniques for separating foreground and background information in uploaded images.

The project therefore demonstrates both:

```text
Real-Time Computer Vision
        +
Classical Image Processing
```

The security monitoring pipeline can:

* 👤 Detect people using YOLOv8n
* 🎯 Maintain persistent tracking IDs with ByteTrack
* 🛡️ Monitor one or more polygon-based security regions
* 🚪 Detect entries and exits
* ⏱️ Record event timestamps
* 🧠 Apply stable-frame debounce to reduce boundary jitter
* 👥 Calculate active people inside each monitored region
* 📄 Generate structured CSV event logs
* 🎥 Produce annotated MP4 output
* 📁 Process multiple videos automatically

The segmentation pipeline can:

* 🖼️ Accept uploaded images
* ⚫ Apply binary thresholding
* 🔆 Apply adaptive thresholding
* 📊 Apply Otsu thresholding
* 🔍 Compare original and segmented images side by side
* 💾 Generate downloadable PNG results

---

## 🎯 Project Objectives

The project was designed around two practical computer vision problems.

### Security Monitoring

A basic object detector can identify a person, but security monitoring requires understanding **where that person is, whether they entered a monitored area, whether they left it, and when the event occurred**.

The system therefore extends object detection into a state-aware monitoring pipeline:

```text
Person Detection
       ↓
Persistent Tracking
       ↓
Centroid Calculation
       ↓
Polygon ROI Evaluation
       ↓
Stable State Detection
       ↓
Entry / Exit Event
       ↓
Timestamped Logging
       ↓
Annotated Video
```

### Image Segmentation

The second component demonstrates how different classical thresholding techniques can transform grayscale images into binary masks.

Rather than assuming that one segmentation method works for every image, the application provides:

```text
Binary Thresholding
Adaptive Thresholding
Otsu Thresholding
```

This makes it possible to compare how different thresholding strategies respond to different image characteristics.

---

# 🧠 System Architecture

```text
                         ┌──────────────────────┐
                         │      Streamlit       │
                         │     Application      │
                         └───────────┬──────────┘
                                     │
                       ┌─────────────┴─────────────┐
                       │                           │
                       ▼                           ▼
              ┌─────────────────┐        ┌─────────────────┐
              │ Security Monitor│        │   Segmentation  │
              │      Tab        │        │       Tab       │
              └────────┬────────┘        └────────┬────────┘
                       │                          │
                       ▼                          ▼
                  Video Input                Image Input
                       │                          │
                       ▼                          ▼
              ┌─────────────────┐          ┌──────────────┐
              │    YOLOv8n      │          │   Grayscale  │
              │ Person Detection│          │ Conversion   │
              └────────┬────────┘          └──────┬───────┘
                       │                          │
                       ▼                    ┌─────┼─────┐
              ┌─────────────────┐            │     │     │
              │    ByteTrack   │         Binary Adaptive Otsu
              │ Persistent IDs │            │     │     │
              └────────┬────────┘            └─────┼─────┘
                       │                            │
                       ▼                            ▼
              Track ID + Centroid          Segmented Images
                       │
                       ▼
                  Polygon ROIs
                       │
                       ▼
             Stable Inside / Outside
                       │
                  ┌────┴────┐
                  ▼         ▼
                ENTRY      EXIT
                  │         │
                  └────┬────┘
                       ▼
                 CSV Event Log
                       +
                 Processed MP4
```

---

# 🛡️ Part A — Intelligent Security Monitoring

## 👤 Person Detection

The security monitoring pipeline uses the **COCO-pretrained YOLOv8n model** and restricts inference to the person class.

```python
model.track(
    frame,
    persist=True,
    tracker="bytetrack.yaml",
    classes=[0],
)
```

The `classes=[0]` parameter selects the COCO **person** class.

This allows the system to focus its computational resources on people rather than processing every object category supported by the model.

---

## 🎯 Persistent Tracking

Object detection operates independently on individual frames.

For security monitoring, however, the system needs to know whether a person detected in one frame is the same person detected in the next frame.

ByteTrack provides this association.

With:

```python
persist=True
```

tracking state is maintained across consecutive frames.

Example:

```text
Frame 1 → ID 12
Frame 2 → ID 12
Frame 3 → ID 12
Frame 4 → ID 12
```

This persistent identity makes it possible to reason about movement over time.

The system also records the unique tracking IDs observed during each processed video.

> **Important:** Tracking IDs represent tracker identities, not guaranteed real-world identities. Occlusion, missed detections, or long disappearances can cause an individual to receive a new ID.

---

# 🛡️ Polygon-Based Security Zones

Security regions are represented as arbitrary polygons containing at least three points.

Example:

```json
[
  {
    "name": "Main Entrance",
    "points": [
      [80, 80],
      [1180, 80],
      [1180, 620],
      [80, 620]
    ]
  }
]
```

The points are specified using pixel coordinates from the original video frame.

Multiple regions can be configured simultaneously:

```json
[
  {
    "name": "Door A",
    "points": [
      [50, 50],
      [400, 50],
      [400, 400],
      [50, 400]
    ]
  },
  {
    "name": "Door B",
    "points": [
      [500, 50],
      [850, 50],
      [850, 400],
      [500, 400]
    ]
  }
]
```

Each monitored region maintains its own state.

The system identifies each person/region combination using:

```text
(track_id, roi_name)
```

This allows the same person to be monitored independently across multiple security zones.

---

# 📍 Centroid-Based Position Analysis

For each tracked person, the system calculates a centroid from the detected bounding box.

Conceptually:

```text
        Bounding Box
      ┌─────────────┐
      │             │
      │      ●      │ ← Centroid
      │             │
      └─────────────┘
```

The centroid provides a compact representation of the person's position and is used to determine whether the person is inside or outside a configured polygon ROI.

---

# 🚪 Entry Detection

A person does not immediately trigger an entry event simply because their centroid briefly touches or enters a monitored region.

Instead, the system uses a **stable-frame debounce mechanism**.

The person must remain inside the configured ROI for the required number of consecutive stable frames.

The default is:

```text
3 stable frames
```

Conceptually:

```text
Outside
   ↓
Inside
   ↓
Inside
   ↓
Inside
   ↓
ENTRY
```

This reduces false events caused by:

* Centroid jitter
* Boundary movement
* Temporary detection noise
* Minor tracking fluctuations

---

# 🚶 Exit Detection

Exit events use the same state-stabilization principle.

A person must remain outside the monitored region for the configured number of stable frames before an exit event is recorded.

```text
Inside
   ↓
Outside
   ↓
Outside
   ↓
Outside
   ↓
EXIT
```

This prevents a person standing near a boundary from generating repeated entry and exit events simply because their centroid moves slightly between frames.

---

# 👥 Active People

A person is considered **active inside an ROI** when their current state for that region is marked as inside.

The processed video can display monitoring information such as:

```text
Active in ROI
Entries
Exits
Unique Track IDs
```

This allows the system to provide both real-time state information and historical event information.

---

# ⏱️ Event Timestamps

Every recorded security event includes the elapsed time within the video.

For example:

```text
0.80 seconds → ENTRY
4.24 seconds → EXIT
```

Using video-relative timestamps makes results reproducible regardless of the actual date or time at which the video was processed.

---

# 📄 Event Logging

Security events are stored in CSV format.

Each event record can contain:

```text
timestamp_s
event
track_id
roi
entry_time_s
exit_time_s
duration_s
frame
status
```

Example:

```text
0.80,ENTRY,12,Main Entrance,0.80,,,20,active
4.24,EXIT,12,Main Entrance,0.80,4.24,3.44,106,inactive
```

This makes the results suitable for further analysis using Python, Excel, pandas, databases, or other analytics tools.

---

# 🎥 Annotated Security Video

The security monitoring pipeline generates an annotated MP4 containing visual information from the analysis.

The output can include:

* Person bounding boxes
* Tracking IDs
* Polygon ROIs
* Active-person counts
* Entry events
* Exit events
* Monitoring statistics

This provides an immediately understandable visual representation of the security analysis.

---

# 📁 Batch Security Monitoring

The system supports processing every supported video in a folder automatically.

Place videos inside:

```text
sample_videos/
```

Then run:

```powershell
python coding_practice/batch_security_monitoring.py --input-dir sample_videos --output-dir outputs/security_monitoring
```

The script automatically discovers and processes the supported videos.

There is no need to execute the command separately for each video.

---

# 📤 Multiple Video Upload

The Streamlit application also supports multiple video uploads.

Users can:

```text
Select multiple videos
        ↓
Upload once
        ↓
Run analysis
        ↓
Process every selected video
        ↓
Generate individual MP4 + CSV outputs
```

Each input video receives its own processed video and event log.

---

# 🖼️ Part B — Image Segmentation

The second part of the project demonstrates classical image segmentation using three thresholding techniques.

```text
Input Image
     ↓
Grayscale Conversion
     ↓
┌─────────────┬─────────────┬─────────────┐
│   Binary    │  Adaptive   │    Otsu     │
│ Threshold   │  Threshold  │  Threshold  │
└─────────────┴─────────────┴─────────────┘
     ↓
Segmented PNG Outputs
```

---

## ⚫ Binary Thresholding

Binary thresholding applies a single global threshold to the image.

Conceptually:

```text
Pixel < threshold     → Black
Pixel ≥ threshold     → White
```

It is particularly useful when the foreground and background have relatively predictable intensity levels.

---

## 🔆 Adaptive Thresholding

Adaptive thresholding calculates a local threshold based on neighboring pixels.

This can make it more effective than a single global threshold when illumination varies across an image.

For example:

```text
Bright Region → Local threshold
Dark Region   → Different local threshold
```

This makes adaptive thresholding useful for images with uneven lighting.

---

## 📊 Otsu Thresholding

Otsu's method automatically selects a global threshold using the image histogram.

Instead of requiring a threshold value to be manually chosen, Otsu attempts to find a threshold that provides effective separation between foreground and background intensity distributions.

---

# 🔬 Comparing Segmentation Methods

There is no universally optimal thresholding method.

| Method       | Best suited for                                          |
| ------------ | -------------------------------------------------------- |
| **Binary**   | Predictable lighting and foreground/background intensity |
| **Adaptive** | Uneven or changing illumination                          |
| **Otsu**     | Images with reasonably distinct intensity groups         |

For a particular dataset, the most appropriate method should be selected based on the quality of the resulting segmentation masks.

The project therefore encourages comparing all three methods rather than assuming one technique will always perform best.

---

# 📁 Output Structure

After processing, outputs are organized by task:

```text
outputs/
│
├── security_monitoring/
│   ├── mall_security_monitor.mp4
│   ├── mall_security_monitor_events.csv
│   ├── office_security_monitor.mp4
│   ├── office_security_monitor_events.csv
│   └── ...
│
└── segmentation/
    ├── sample_binary.png
    ├── sample_adaptive.png
    └── sample_otsu.png
```

This separation keeps video analytics and image-processing outputs organized and easy to inspect.

---

# 📂 Project Structure

```text
Intelligent-Security-Monitoring/
│
├── app.py
│   └── Streamlit application
│
├── monitoring.py
│   └── YOLO detection, ByteTrack tracking,
│       polygon ROI monitoring, and event logic
│
├── segmentation.py
│   └── Binary, Adaptive, and Otsu segmentation
│
├── requirements.txt
│   └── Python dependencies
│
├── README.md
│   └── Project documentation
│
├── coding_practice/
│   ├── batch_security_monitoring.py
│   │   └── Batch security-video processing
│   │
│   └── batch_segmentation.py
│       └── Batch image segmentation
│
├── sample_videos/
│   └── Sample security-monitoring videos
│
├── sample_input_images/
│   └── Sample images for segmentation
│
├── outputs/
│   ├── security_monitoring/
│   └── segmentation/
│
├── screenshots/
│   └── Application screenshots
│
└── .streamlit/
    └── config.toml
```

---

# 🛠️ Technology Stack

| Technology                     | Purpose                             |
| ------------------------------ | ----------------------------------- |
| **Python 3.12**                | Core programming language           |
| **YOLOv8n**                    | Person detection                    |
| **Ultralytics**                | YOLO model implementation           |
| **ByteTrack**                  | Multi-object tracking               |
| **OpenCV**                     | Video and image processing          |
| **NumPy**                      | Numerical and coordinate operations |
| **Streamlit**                  | Interactive web application         |
| **ngrok**                      | Public local deployment             |
| **CSV**                        | Event logging                       |
| **Pytest / Python validation** | Testing and validation              |
| **Git & GitHub**               | Version control                     |

---

# 🚀 Local Setup

Python 3.12 is recommended.

Create a virtual environment:

```powershell
py -3.12 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install project dependencies:

```powershell
pip install -r requirements.txt
```

---

# 🧪 Syntax Verification

Before launching the application, the main Python modules can be checked for syntax errors:

```powershell
python -m py_compile app.py
python -m py_compile monitoring.py
python -m py_compile segmentation.py
python -m py_compile coding_practice/batch_security_monitoring.py
python -m py_compile coding_practice/batch_segmentation.py
```

A successful compilation indicates that the files can be parsed correctly by Python.

---

# ▶️ Running the Application

Start Streamlit:

```powershell
streamlit run app.py
```

The application will normally become available at:

```text
http://localhost:8501
```

The Streamlit interface provides separate workflows for security monitoring and image segmentation.

---

# 💻 Batch Testing Without Streamlit

## Security Monitoring

```powershell
python coding_practice/batch_security_monitoring.py --input-dir sample_videos --output-dir outputs/security_monitoring
```

The command automatically processes every supported video in the input directory.

## Image Segmentation

```powershell
python coding_practice/batch_segmentation.py --input-dir sample_input_images --output-dir outputs/segmentation
```

Each input image is processed using:

```text
Binary
Adaptive
Otsu
```

The resulting PNG files are written to the output directory.

---

# 🌐 Public Deployment with ngrok

The application can be exposed publicly from a local machine using ngrok.

First start Streamlit:

```powershell
streamlit run app.py
```

Then open a second terminal and run:

```powershell
ngrok http 8501
```

ngrok will provide an HTTPS forwarding address that can be used to access the locally running Streamlit application remotely.

> **Security note:** Never commit your ngrok authentication token or other private credentials to GitHub.

---

# ☁️ Streamlit Community Cloud Compatibility

The project is structured to support Streamlit Community Cloud deployment.

The primary application files are:

```text
app.py
requirements.txt
monitoring.py
segmentation.py
```

The segmentation workflow is lightweight and well suited to hosted environments.

The security-monitoring workflow performs frame-by-frame:

```text
YOLO inference
        +
ByteTrack tracking
        +
Polygon analysis
        +
Event management
```

As a result, long videos can require significant CPU, RAM, and processing time.

For public demonstrations, short and clear video clips are recommended.

For large collections of videos, local batch processing is generally more appropriate.

---

# ⚠️ Challenges and Limitations

## Occlusion

People may overlap with one another or temporarily disappear from view. This can cause missed detections or tracking ID changes.

## ID Switches

ByteTrack improves identity continuity, but heavy occlusion, long disappearances, and crowded scenes can still result in ID switches.

## Camera Movement

Polygon ROIs use fixed pixel coordinates.

A moving camera can therefore make a previously valid security region inaccurate.

## ROI Boundary Jitter

A person's centroid may repeatedly move near the boundary of a polygon.

Stable-frame debounce reduces this problem but cannot completely eliminate it.

## Tracking Identity

Tracking IDs represent the state maintained by the tracker.

They should not be interpreted as guaranteed identities of real-world individuals.

## Small or Distant People

The lightweight YOLOv8n model may have difficulty detecting very small, distant, blurred, or partially visible people.

## Classical Segmentation

Binary, adaptive, and Otsu thresholding are intensity-based image-processing methods.

They are not semantic segmentation models and therefore do not provide object-level understanding.

## CPU Processing

YOLO + ByteTrack video processing can be computationally expensive on CPU-only environments, particularly for long or high-resolution videos.

---

# 🔮 Future Improvements

Potential future enhancements include:

* Multiple advanced security-zone configurations
* Camera-motion compensation
* Perspective-aware ROI analysis
* Real-time CCTV integration
* Face anonymization
* Advanced crowd analytics
* Real-time security alerts
* Email or notification integration
* Database-backed event storage
* JSON event exports
* Event dashboards
* Heatmap generation
* Multi-camera tracking
* GPU-accelerated inference
* Deep-learning-based semantic segmentation
* Instance segmentation
* Automatic ROI configuration
* Advanced anomaly detection

---

# 🔐 Privacy and Responsible Use

Security-monitoring systems can process video containing identifiable individuals.

When using real-world footage:

* Obtain appropriate authorization.
* Follow applicable privacy regulations.
* Use legally authorized datasets.
* Avoid unnecessary storage of raw footage.
* Restrict access to generated outputs.
* Apply anonymization where appropriate.
* Do not use tracking IDs as proof of real-world identity.

The system should be deployed responsibly and according to the legal and organizational requirements of its intended environment.

---

# 📚 Core Concepts Demonstrated

This project demonstrates practical implementation of:

* Computer Vision
* Artificial Intelligence
* Object Detection
* YOLOv8
* Multi-Object Tracking
* ByteTrack
* Polygon Geometry
* Region-of-Interest Analysis
* Centroid Tracking
* Event Detection
* State Management
* Entry and Exit Detection
* Debouncing
* Timestamped Event Logging
* CSV Data Generation
* Video Processing
* Image Segmentation
* Binary Thresholding
* Adaptive Thresholding
* Otsu Thresholding
* Streamlit Development
* Local Web Deployment
* ngrok
* Batch Processing
* Modular Python Architecture

---

# 🧩 Design Philosophy

The project separates detection, tracking, monitoring, segmentation, and presentation into independent components.

The security workflow follows:

```text
Detection
    ↓
Tracking
    ↓
Position Analysis
    ↓
ROI Evaluation
    ↓
State Stabilization
    ↓
Event Detection
    ↓
Event Logging
    ↓
Video Annotation
```

The segmentation workflow follows:

```text
Input Image
    ↓
Grayscale
    ↓
Thresholding
    ↓
Binary Mask
    ↓
Visual Comparison
    ↓
Downloadable Output
```

This modular design improves:

* Maintainability
* Debugging
* Testing
* Reusability
* Scalability
* Future model integration

---

# 📋 Final Validation Checklist

Before considering the project ready for demonstration or deployment:

```text
[ ] Add 5+ legally usable sample videos
[ ] Add sample input images
[ ] Run batch security monitoring
[ ] Verify MP4 outputs
[ ] Verify CSV event logs
[ ] Run batch segmentation
[ ] Verify Binary outputs
[ ] Verify Adaptive outputs
[ ] Verify Otsu outputs
[ ] Run Streamlit locally
[ ] Test image upload
[ ] Test video upload
[ ] Test multiple video upload
[ ] Test sample-folder batch processing
[ ] Test download functionality
[ ] Test security ROIs
[ ] Test entry detection
[ ] Test exit detection
[ ] Test debounce behavior
[ ] Test ngrok deployment
[ ] Capture demonstration screenshots
[ ] Verify GitHub repository contents
```

---

# 📄 License

This project is provided for educational, research, and development purposes.

Before deploying the system commercially or in a real security environment, review the licensing requirements of the software libraries, model weights, datasets, sample footage, and deployment services being used.

---

# 👨‍💻 Author

## Hadeed Jalani

AI and Machine Learning Developer focused on building practical solutions in:

```text
Artificial Intelligence
Machine Learning
Deep Learning
Computer Vision
Python
Object Detection
Object Tracking
Image Processing
Data Analytics
```
---

<p align="center">
  <strong>🛡️ Intelligent Security Monitoring + Image Segmentation</strong>
  <br>
  Detection • Tracking • Security Zones • Event Analytics • Segmentation
  <br><br>
  <strong>Built with Python, YOLOv8n, ByteTrack, OpenCV & Streamlit</strong>
</p>
