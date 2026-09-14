# Day 40 Smart Video Analytics System

## 1. Project Overview

The **Smart Video Analytics System** is an AI-powered video analysis application developed as part of **Day 40** of the MLB Internship project.

The purpose of this project is to process recorded videos frame by frame, detect and track moving objects, monitor a user-defined region of interest, identify entry and exit events, calculate real-time performance statistics, and generate useful analytical reports.

The system uses:

* **YOLOv8n** for object detection
* **ByteTrack** or **BoT-SORT** for object tracking
* **OpenCV** for video processing and visualization
* **Streamlit** for the interactive user interface
* **Pandas** for event logging and analytics
* **Python** for the complete processing pipeline

The application is designed for recorded videos rather than live camera input. It can analyze videos containing people, vehicles, or other moving objects, provided the selected YOLO model is trained to detect those object categories.

---

## 2. Project Objective

The main objective of this project is to understand and implement a complete video analytics pipeline:

```text
Video
  ↓
Frame-by-Frame Processing
  ↓
YOLO Object Detection
  ↓
Object Tracking
  ↓
Tracking IDs
  ↓
Region of Interest Monitoring
  ↓
Entry and Exit Detection
  ↓
Analytics and FPS Calculation
  ↓
Processed Video + events.csv
```

The project focuses on creating a reliable, understandable, and practical video analytics application rather than adding unnecessary complexity.

---

## 3. Main Features

The Smart Video Analytics System provides the following features:

### 3.1 Video Upload

The application accepts recorded video files through the Streamlit interface.

Supported formats include:

* `.mp4`
* `.avi`
* `.mov`
* `.mkv`

The recommended input videos are approximately **15–30 seconds long**.

The videos should contain moving objects such as:

* People walking
* Vehicles moving through a road
* Objects entering or leaving a defined area
* Crowds moving through a monitored space

---

### 3.2 YOLO Object Detection

The application uses **YOLOv8n**, a lightweight version of the YOLO object-detection model.

YOLO detects objects in each processed frame and returns:

* Bounding-box coordinates
* Object class
* Confidence score
* Detection location

For the current implementation, the application focuses on detecting **people** using the person class.

The model is configured with a confidence threshold and an IoU threshold to control detection quality.

---

### 3.3 Object Tracking

Object detection identifies objects in individual frames, but it does not automatically maintain the identity of an object across multiple frames.

To solve this problem, the application uses object tracking.

The tracker assigns a unique numerical ID to each detected object. For example:

```text
Person A → ID 1
Person B → ID 2
Person C → ID 3
```

When the same person appears in subsequent frames, the tracker attempts to preserve the same ID.

The application supports:

* **ByteTrack**
* **BoT-SORT**

#### ByteTrack

ByteTrack is a lightweight tracking algorithm that is useful when processing speed is important.

#### BoT-SORT

BoT-SORT can provide more robust tracking in some difficult scenes, especially when objects experience partial occlusion or temporary detection changes.

Tracking performance depends on:

* Video quality
* Object size
* Lighting
* Camera movement
* Object occlusion
* Frame skipping
* Detection confidence
* Tracker configuration

---

## 4. Region of Interest

A **Region of Interest**, commonly called an ROI, is a selected area of the video frame that the system monitors.

The ROI allows the application to focus on a particular area instead of treating the entire frame as the monitored region.

Examples include:

* A building entrance
* A parking area
* A road section
* A doorway
* A restricted zone
* A corridor
* A shop entrance

The current application uses a rectangular ROI. The user can adjust the ROI using sliders in the Streamlit sidebar.

The ROI is defined using normalized coordinates:

```text
ROI left
ROI top
ROI right
ROI bottom
```

These values are converted into pixel coordinates according to the width and height of the input video.

For example:

```text
ROI left   = 0.20
ROI top    = 0.20
ROI right  = 0.80
ROI bottom = 0.85
```

This represents a rectangle covering the central portion of the video frame.

---

## 5. Object Position and ROI Membership

For every detected object, the application calculates the center point of its bounding box.

If the bounding box is represented as:

```text
x1, y1, x2, y2
```

the center point is calculated as:

```text
center_x = (x1 + x2) / 2
center_y = (y1 + y2) / 2
```

The application then checks whether the center point is inside the ROI.

The object is considered inside the ROI when:

```text
ROI_left  ≤ center_x ≤ ROI_right
ROI_top   ≤ center_y ≤ ROI_bottom
```

This approach provides a simple and understandable way to determine whether an object has entered or left the monitored region.

---

## 6. Entry and Exit Detection

The application detects entry and exit events by comparing the current ROI status of a tracking ID with its previous ROI status.

The main logic is:

```text
Outside → Inside = ENTRY
Inside → Outside = EXIT
```

For example:

```text
Frame 1: ID 7 is outside the ROI
Frame 2: ID 7 is outside the ROI
Frame 3: ID 7 is inside the ROI
         → ENTRY event is recorded

Frame 4: ID 7 is inside the ROI
Frame 5: ID 7 is outside the ROI
         → EXIT event is recorded
```

Each event contains information such as:

* Tracking ID
* Event type
* Frame number
* Timestamp
* Entry time
* Exit time
* Duration
* Event status

The application also handles objects that remain inside the ROI until the video ends. These sessions are closed using the final video timestamp.

---

## 7. Tracking IDs

Tracking IDs are important because they allow the application to distinguish between different objects.

For example, if three people appear in a video, the tracker may assign:

```text
Person 1 → ID 4
Person 2 → ID 8
Person 3 → ID 12
```

The IDs are used to:

* Count unique objects
* Track movement across frames
* Monitor ROI membership
* Detect entry events
* Detect exit events
* Calculate dwell time
* Reduce duplicate event records
* Associate events with a specific tracked object

Tracking IDs are not permanent identities. A tracking ID belongs to the current video-processing session and may change if the tracker loses an object.

---

## 8. Analytics Generated by the Application

The application calculates several statistics during video processing.

### 8.1 Total Objects

This is the number of unique tracking IDs observed during the video.

For example:

```text
Total Objects: 18
```

This does not necessarily mean that 18 objects were visible at the same time. It means that 18 unique tracked objects were observed during the processing session.

---

### 8.2 Current ROI Count

This represents the number of tracked objects currently inside the ROI.

The value can change from frame to frame as objects enter or leave the monitored area.

---

### 8.3 Total Entries

This is the total number of detected transitions from outside the ROI to inside the ROI.

```text
Outside → Inside
```

---

### 8.4 Total Exits

This is the total number of detected transitions from inside the ROI to outside the ROI.

```text
Inside → Outside
```

---

### 8.5 Maximum Objects in ROI

This represents the highest number of tracked objects present inside the ROI at any point during the video.

For example:

```text
Maximum Objects in ROI: 7
```

This can be useful for estimating crowd density or peak occupancy.

---

### 8.6 Average Objects in ROI

This is the average number of objects detected inside the ROI during processed frames.

It provides an overall estimate of how occupied the monitored region was throughout the video.

---

### 8.7 Average Dwell Time

Dwell time is the amount of time an object remains inside the ROI.

The calculation is:

```text
Dwell Time = Exit Time − Entry Time
```

The application calculates the average dwell time for completed ROI sessions.

---

### 8.8 Average FPS

FPS means **frames per second**.

The application calculates the actual processing speed using the number of processed frames divided by the total processing time.

```text
Average FPS = Processed Frames / Processing Time
```

This is the measured processing speed of the application and may differ from the original video's FPS.

---

### 8.9 Processing Time

Processing time represents the amount of time required to analyze the complete input video.

This value depends on:

* Computer hardware
* CPU or GPU availability
* Input resolution
* Model size
* Detection settings
* Tracker selection
* Frame skipping
* Video complexity

---

## 9. Performance Optimization

The application includes several controls for improving processing speed.

### 9.1 Image Size

The application supports two inference image sizes:

* `480`
* `640`

A smaller image size usually requires less computation and may increase FPS.

However, reducing image size can also make small objects more difficult to detect.

#### 480px

Advantages:

* Faster inference
* Lower computational cost
* Useful for quick testing

Disadvantages:

* Small objects may be missed
* Bounding boxes may be less precise

#### 640px

Advantages:

* Better detection detail
* More suitable for smaller objects
* Often more accurate in complex scenes

Disadvantages:

* Higher computational cost
* Potentially lower FPS

---

### 9.2 Frame Skipping

Frame skipping reduces the number of frames sent to the detection and tracking model.

For example:

```text
Frame skip = 1
```

means that every frame is processed.

```text
Frame skip = 2
```

means that approximately every second frame is processed.

Frame skipping can improve speed, but excessive skipping may cause:

* Missed short events
* Less stable tracking
* More ID switches
* Inaccurate entry and exit timing
* Objects moving significantly between processed frames

For this reason, frame skipping should be tested carefully.

---

### 9.3 Confidence Threshold

The confidence threshold controls how confident YOLO must be before accepting a detection.

A higher confidence threshold:

* Reduces weak detections
* May reduce false positives
* Can miss partially visible objects

A lower confidence threshold:

* Accepts more possible detections
* May improve recall
* Can introduce false positives

The default value is configured for balanced performance and can be adjusted through the interface.

---

### 9.4 IoU Threshold

IoU means **Intersection over Union**.

It measures the overlap between two bounding boxes.

The IoU threshold influences how overlapping detections are filtered during non-maximum suppression.

A lower or higher IoU value can affect:

* Duplicate detections
* Overlapping objects
* Crowded scenes
* Detection stability

The best value depends on the video and object density.

---

### 9.5 Tracker Selection

The application supports two tracking options:

```text
ByteTrack
BoT-SORT
```

ByteTrack is generally a useful starting point when processing speed is important.

BoT-SORT can be tested for scenes with:

* Occlusion
* Crowded areas
* Frequent object overlap
* More complex movement

The best tracker should be selected based on both tracking quality and processing speed.

---

### 9.6 Tracking Trails

The application optionally displays tracking trails.

A tracking trail shows the recent movement path of an object using its center-point history.

Trails can help visualize movement, but disabling them may slightly reduce drawing overhead.

---

## 10. Performance Benchmark

The project includes a performance benchmarking script.

The benchmark compares the following configurations:

| Configuration   | Image Size | Frame Skip |
| --------------- | ---------: | ---------: |
| Configuration 1 |      640px |          1 |
| Configuration 2 |      480px |          1 |
| Configuration 3 |      640px |          2 |
| Configuration 4 |      480px |          2 |

Run the benchmark using:

```powershell
python performance_test.py --video sample_videos/video_01.mp4
```

The results are saved to:

```text
outputs/performance_comparison.csv
```

The benchmark records:

* Video name
* Image size
* Frame skip
* Average FPS
* Processing time
* Total objects
* Total entries
* Total exits
* Maximum ROI count

The best configuration should be selected using the actual measured results.

A configuration with the highest FPS is not automatically the best overall configuration. Detection accuracy and tracking stability must also be considered.

For example:

* A 480px configuration may be faster.
* A 640px configuration may detect small objects more reliably.
* Frame skipping may improve speed but reduce event accuracy.
* BoT-SORT may improve tracking in difficult scenes but require more processing.

---

## 11. Output Files

After processing a video, the application generates the following files.

### 11.1 Processed Video

The processed video contains:

* Bounding boxes
* Tracking IDs
* ROI rectangle
* Current ROI count
* Unique object count
* FPS
* Frame number
* Optional tracking trails

The output is saved in the `outputs` directory.

Example:

```text
outputs/video_01_processed.mp4
```

---

### 11.2 events.csv

The `events.csv` file contains the detected ROI events.

The main columns are:

| Column         | Description                           |
| -------------- | ------------------------------------- |
| `track_id`     | Unique tracking ID                    |
| `event`        | `ENTRY` or `EXIT`                     |
| `frame`        | Frame number where the event occurred |
| `timestamp_s`  | Event timestamp in seconds            |
| `entry_time_s` | Entry timestamp                       |
| `exit_time_s`  | Exit timestamp                        |
| `duration_s`   | Time spent inside the ROI             |
| `status`       | Event or session status               |

Example structure:

```text
track_id,event,frame,timestamp_s,entry_time_s,exit_time_s,duration_s,status
4,ENTRY,120,4.80,4.80,,,active
4,EXIT,245,9.80,4.80,9.80,5.00,completed
```

The actual values depend on the input video and tracking results.

---

### 11.3 performance_comparison.csv

This file contains the results of the performance benchmark.

It can be used to compare the processing configurations and identify the most suitable setup.

---

## 12. Project Structure

```text
Day-40/
│
├── app.py
├── video_analytics.py
├── performance_test.py
├── coding_practice.py
├── test_day40.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── .streamlit/
│   └── config.toml
│
├── sample_videos/
│   ├── video_01.mp4
│   ├── video_02.mp4
│   ├── video_03.mp4
│   └── README.md
│
├── outputs/
│   ├── video_01_processed.mp4
│   ├── video_02_processed.mp4
│   ├── video_03_processed.mp4
│   ├── events.csv
│   ├── performance_comparison.csv
│   └── README.md
│
└── screenshots/
    └── README.md
```

### File Descriptions

#### `app.py`

The main Streamlit application.

It provides:

* Video upload
* Processing controls
* ROI controls
* Tracker selection
* Image-size selection
* Frame-skipping controls
* Progress updates
* Analytics metrics
* Processed-video preview
* CSV download

#### `video_analytics.py`

Contains the main video-processing logic.

It is responsible for:

* Loading YOLO
* Reading video frames
* Running object detection
* Running object tracking
* Calculating object centers
* Checking ROI membership
* Detecting entries and exits
* Calculating statistics
* Writing processed videos
* Exporting `events.csv`

#### `performance_test.py`

Runs the performance comparison across different image sizes and frame-skipping configurations.

#### `coding_practice.py`

Provides a basic OpenCV-based practice script for learning:

* YOLO tracking
* Tracking IDs
* FPS
* ROI drawing
* Object counting
* Entry and exit detection

#### `test_day40.py`

Contains basic tests for:

* ROI construction
* ROI membership
* Event dataframe normalization
* Empty event data handling

#### `requirements.txt`

Lists the Python dependencies required by the project.

#### `sample_videos/`

Stores the three short input videos used for testing.

#### `outputs/`

Stores processed videos, event logs, and benchmark results.

#### `screenshots/`

Stores screenshots captured for documentation and submission.

---

## 13. Installation

### Step 1: Open the Project Directory

Open PowerShell inside the `Day-40` folder.

```powershell
cd path\to\Day-40
```

---

### Step 2: Create a Virtual Environment

```powershell
py -3.13 -m venv .venv
```

---

### Step 3: Activate the Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

### Step 4: Upgrade pip

```powershell
python -m pip install --upgrade pip
```

---

### Step 5: Install Dependencies

```powershell
pip install -r requirements.txt
```

The YOLO model may automatically download `yolov8n.pt` the first time the application runs.

---

## 14. Running the Streamlit Application

Start the application using:

```powershell
streamlit run app.py
```

Streamlit will display a local address similar to:

```text
http://localhost:8501
```

If port `8501` is already in use, run:

```powershell
streamlit run app.py --server.port 8502
```

Then open the displayed local URL in a browser.

---

## 15. Application Workflow

The recommended workflow is:

1. Open the Streamlit application.
2. Upload a short video.
3. Select the confidence threshold.
4. Select the IoU threshold.
5. Choose the inference image size.
6. Select the frame-skipping value.
7. Choose ByteTrack or BoT-SORT.
8. Adjust the ROI sliders.
9. Enable or disable tracking trails.
10. Click **Start Video Analytics**.
11. Wait for processing to complete.
12. Review the analytics summary.
13. Watch the processed video.
14. Download the processed video.
15. Download `events.csv`.
16. Review the event table.
17. Repeat the process for the remaining test videos.

---

## 16. Coding Practice

The coding-practice script is intended to reinforce the core concepts of real-time video analytics.

Place a video at:

```text
sample_videos/video_01.mp4
```

Then run:

```powershell
python coding_practice.py
```

The script displays:

* Bounding boxes
* Tracking IDs
* ROI rectangle
* Current ROI count
* Unique object count
* Entry count
* Exit count
* FPS

Press the `Q` key to close the OpenCV preview window.

---

## 17. Testing

Run the project tests using:

```powershell
python test_day40.py
```

The tests validate important utility functions such as:

* ROI coordinate conversion
* ROI point membership
* Event dataframe formatting
* Empty event dataframe creation

Expected output:

```text
Day 40 tests passed successfully.
```

These tests do not replace full video testing. The application must still be tested with real videos to evaluate:

* Detection quality
* Tracking stability
* ROI accuracy
* Entry and exit correctness
* FPS
* Output-video generation

---

## 18. Dataset and Video Testing

The project requires three short videos.

Each video should be approximately:

```text
15–30 seconds
```

Recommended sources include:

* Self-recorded videos
* Pexels
* Pixabay
* Other sources providing suitable usage rights

The three videos should ideally represent different situations.

### Suggested Test Cases

#### Video 1 — Normal Movement

Objects move through the ROI with limited overlap.

Purpose:

* Validate basic detection
* Validate tracking IDs
* Validate entry and exit events

#### Video 2 — Crossing Objects

Two or more objects cross paths.

Purpose:

* Test tracking-ID consistency
* Observe possible ID switches
* Compare ByteTrack and BoT-SORT

#### Video 3 — Crowded or Complex Scene

Several objects appear together or partially block one another.

Purpose:

* Test object counting
* Test maximum ROI occupancy
* Evaluate detection confidence
* Evaluate tracker stability

The application should be tested on all three videos.

---

## 19. Problems and Solutions

### Problem 1: Low Processing FPS

**Cause:**

* Large inference image size
* Complex video
* CPU-only processing
* Too many objects
* Expensive tracking configuration

**Possible solutions:**

* Use 480px inference
* Enable frame skipping
* Disable tracking trails
* Use ByteTrack
* Use shorter videos
* Reduce unnecessary drawing operations

---

### Problem 2: Missed Objects

**Cause:**

* Objects are too small
* Confidence threshold is too high
* Video quality is poor
* Objects are partially hidden
* Inference image size is too small

**Possible solutions:**

* Increase image size to 640px
* Lower the confidence threshold slightly
* Improve input-video quality
* Reduce frame skipping
* Test the video with different settings

---

### Problem 3: Tracking ID Switches

**Cause:**

* Objects crossing each other
* Severe occlusion
* Sudden movement
* Low frame rate
* Excessive frame skipping

**Possible solutions:**

* Test BoT-SORT
* Reduce frame skipping
* Use higher-quality footage
* Increase inference image size
* Avoid extremely crowded scenes for basic testing

---

### Problem 4: Incorrect Entry or Exit Events

**Cause:**

* ROI is poorly positioned
* Bounding-box center is near the ROI boundary
* Tracking IDs change
* Frame skipping causes an object to move across the ROI quickly

**Possible solutions:**

* Adjust the ROI boundaries
* Use frame skip `1`
* Test a larger ROI
* Improve tracking stability
* Use videos with clear movement through the region

---

### Problem 5: Duplicate Events

**Cause:**

* An object repeatedly moves across the ROI boundary
* Tracking instability
* Bounding-box center fluctuates near the ROI edge

**Possible solutions:**

* Use a clearly defined ROI
* Reduce detection noise
* Improve tracker settings
* Avoid placing the ROI boundary directly on a highly congested area
* Add event debouncing or boundary tolerance in future improvements

---

### Problem 6: Output Video Cannot Be Opened

**Possible causes:**

* Unsupported video codec
* Missing OpenCV installation
* Invalid input video
* Output directory permission issue

**Possible solutions:**

* Confirm that OpenCV is installed
* Use `.mp4` input
* Check that the `outputs` folder exists
* Test with another video
* Verify that the video writer opens successfully

---

## 20. Limitations

The current system has several limitations:

1. It is designed primarily for recorded videos.
2. The default implementation focuses on person detection.
3. Tracking IDs are not permanent identities.
4. Severe occlusion may cause ID switches.
5. Frame skipping can reduce event accuracy.
6. ROI membership is based on the center point of the bounding box.
7. A rectangular ROI is used instead of a polygonal ROI.
8. Detection and tracking quality depends on video conditions.
9. Processing speed depends heavily on available hardware.
10. The application does not identify people by name.
11. The system does not provide facial recognition.
12. The system does not guarantee perfect counting or event detection.

---

## 21. Future Improvements

Possible future improvements include:

* Polygon-based ROI selection
* Multiple ROIs
* Vehicle and object-class selection
* Event debouncing
* Line-crossing detection
* Improved session management
* Real-time charts
* More detailed performance graphs
* GPU acceleration
* Model selection
* Custom-trained YOLO models
* Automatic report generation
* Database storage
* Alert notifications
* Advanced tracking configuration
* Better handling of temporary occlusions
* Batch processing for multiple videos
* Cloud deployment
* Live camera support

---

## 22. Ethical and Responsible Use

This project is intended for educational and analytical purposes.

Users should:

* Use videos legally and responsibly.
* Prefer self-recorded or properly licensed footage.
* Avoid collecting unnecessary personal information.
* Avoid using the system for unauthorized surveillance.
* Avoid making sensitive decisions about individuals based only on automated tracking.
* Understand that tracking IDs are temporary technical identifiers.
* Review results manually before making important decisions.

The application detects and tracks visible objects. It does not determine a person's identity, intent, or behavior beyond the movement patterns represented in the video.

---

## 23. Final Deliverables

The completed Day 40 submission should include:

* Source code
* `app.py`
* `video_analytics.py`
* `performance_test.py`
* `coding_practice.py`
* `test_day40.py`
* `requirements.txt`
* `README.md`
* Three sample videos
* Three processed output videos
* `events.csv`
* `performance_comparison.csv`
* Screenshots
* GitHub repository link
* Active Streamlit URL
* Demonstration video

Before submission, confirm that:

* The application runs successfully.
* All three videos have been tested.
* The generated output videos can be opened.
* `events.csv` contains actual event data.
* Performance results are based on real measurements.
* The GitHub repository contains the required files.
* The deployed Streamlit URL is active.

---

## 24. Conclusion

The Day 40 Smart Video Analytics System demonstrates a complete AI-based video-processing workflow.

It combines object detection, object tracking, ROI monitoring, event detection, performance measurement, and report generation into one application.

The project provides practical experience with:

* Frame-by-frame video processing
* YOLO object detection
* Tracking IDs
* ByteTrack and BoT-SORT
* ROI-based analytics
* Entry and exit detection
* FPS measurement
* Frame skipping
* Performance optimization
* CSV event logging
* Streamlit application development

The final system follows the complete pipeline:

```text
Video
→ Frame Processing
→ YOLO Detection
→ Object Tracking
→ ROI Monitoring
→ Entry/Exit Events
→ Analytics
→ Processed Video
→ events.csv
```

This project forms a foundation for more advanced applications such as smart surveillance, traffic monitoring, crowd analytics, retail analytics, parking monitoring, and automated video intelligence systems.
