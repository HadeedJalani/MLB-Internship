Markdown

````
# Day 39 — Optimized Security Monitoring System

> **An optimized computer-vision pipeline for real-time-style security monitoring, person tracking, region-of-interest analysis, and entry/exit event reporting using YOLOv8 and Ultralytics tracking.**

---

## 📌 Project Overview

Day 39 builds upon the **Day 38 Security Monitoring project** and focuses on improving its performance, configurability, tracking reliability, reporting, and user experience.

The system uses **YOLOv8n** for person detection and Ultralytics-supported tracking algorithms such as **ByteTrack** and **BoT-SORT** to monitor people moving through a defined region of interest.

Instead of only detecting people in individual frames, this project maintains object identities across video frames and records meaningful security events, including:

- When a person enters the monitoring region.
- When a person exits the monitoring region.
- How long a person remains inside the region.
- How many unique people were tracked.
- The peak number of people inside the region.
- The average number of people inside the region.
- The processing speed and number of processed frames.

The application provides a Streamlit interface for uploading images and videos, adjusting inference parameters, viewing processed results, and downloading generated reports.

---

## 🎯 Objectives

The main objectives of Day 39 are to optimize and improve the previous security-monitoring pipeline by introducing:

1. **Configurable detection thresholds**
2. **Inference-speed optimization**
3. **Improved tracking configuration**
4. **ROI-based monitoring**
5. **Persistent tracking IDs**
6. **Entry and exit event detection**
7. **Dwell-time analysis**
8. **Tracking trails**
9. **CSV event reporting**
10. **Image and video support**
11. **Better error handling**
12. **A cleaner and more professional interface**
13. **Automated testing**
14. **Crossing-case tracking evaluation**

---

## 🚀 Main Features

### 1. Person Detection

The system uses **YOLOv8n** to detect people in uploaded images and videos.

Only the `person` class is processed:

```python
classes=[0]
````

In the COCO dataset, class `0` represents a person.

The system draws bounding boxes around detected people and displays their tracking IDs during video processing.

### 2. Persistent Object Tracking

For video input, the system uses Ultralytics tracking functionality to maintain identities across frames.

Supported tracking algorithms:

* ByteTrack

* BoT-SORT

Each detected person receives a tracking ID, for example:

```
ID 1
ID 2
ID 3
```

The tracking ID helps the system distinguish between different people and calculate entry, exit, and dwell-time events.

### 3. Region of Interest Monitoring

The system supports monitoring a specific rectangular region called the Region of Interest, or ROI.

The ROI allows the application to focus on a particular area instead of treating the entire video frame as the monitoring zone.

For example, the ROI could represent:

* A building entrance

* A restricted hallway

* A warehouse section

* A parking entrance

* A security checkpoint

* A laboratory entrance

* A private office area

A person is considered to be inside the ROI when the center point of their bounding box falls within the configured region.

The ROI follows this format:

Python

Run

```
(x1, y1, x2, y2)
```

Where:

* `x1` is the left coordinate.

* `y1` is the top coordinate.

* `x2` is the right coordinate.

* `y2` is the bottom coordinate.

### 4. Entry and Exit Event Detection

The application tracks whether each person is currently inside or outside the ROI.

An `ENTRY` event is generated when a person changes from outside the ROI to inside the ROI.

An `EXIT` event is generated when a person changes from inside the ROI to outside the ROI.

Example event sequence:

```
Person ID 4 → ENTRY
Person ID 4 → EXIT
```

Each event includes information such as:

* Track ID

* Event type

* ROI name

* Frame number

* Timestamp

* Entry time

* Exit time

* Duration

* Event status

This makes the system more useful for security monitoring than a simple people detector.

### 5. Dwell-Time Calculation

Dwell time represents how long a person remains inside the monitoring region.

The system calculates dwell time using:

```
Dwell time = Exit timestamp − Entry timestamp
```

For example:

```
Entry time: 12.50 seconds
Exit time: 20.75 seconds

Dwell time: 8.25 seconds
```

Dwell-time information can help identify:

* Long stays in restricted areas

* Unusual waiting behavior

* People remaining inside a zone for extended periods

* Potential security incidents requiring further review

### 6. Confidence Threshold Control

The Streamlit sidebar provides a confidence threshold slider.

The confidence threshold determines the minimum confidence required for a detection to be considered valid.

A higher confidence threshold generally produces fewer weak detections, but it may also miss people who are:

* Far from the camera

* Partially occluded

* In poor lighting

* Blurred

* Small in the frame

A lower confidence threshold may detect more people, but it can also increase false positives.

The default value is:

```
0.45
```

Recommended starting range:

```
0.35 – 0.60
```

### 7. IoU Threshold Control

The IoU threshold is used during non-maximum suppression.

IoU stands for Intersection over Union.

It measures the overlap between bounding boxes.

The IoU threshold helps determine when overlapping detections should be treated as duplicate detections.

The application allows the user to adjust this value through the Streamlit sidebar.

The default value is:

```
0.50
```

### 8. Inference Image-Size Control

The application allows the user to select the inference image size:

```
320
416
512
640
```

The inference image size affects the balance between speed and detection quality.

|
Image Size

|

Typical Behavior

|
| --- | --- |
|

320

|

Faster processing, lower detail

|
|

416

|

Good speed for many videos

|
|

512

|

Balanced default option

|
|

640

|

More detail, potentially slower processing

|

Smaller image sizes may be useful for:

* Long videos

* CPU-based systems

* Real-time-style demonstrations

* Low-resolution input footage

Larger image sizes may be useful when people appear small in the frame.

### 9. Frame-Skip Optimization

Video processing can be expensive when every frame is analyzed.

The application supports frame skipping through the following control:

```
Process every Nth frame
```

For example:

```
frame_skip = 1
```

Processes every frame.

```
frame_skip = 2
```

Processes every second frame.

```
frame_skip = 3
```

Processes every third frame.

Frame skipping can improve processing speed, especially for long videos.

However, excessive frame skipping may reduce:

* Tracking accuracy

* Entry/exit timing precision

* Detection of short events

* ID consistency during fast movement

For the most accurate results, use:

```
frame_skip = 1
```

For faster demonstrations, try:

```
frame_skip = 2
```

### 10. Tracking Trails

The application can display the recent movement path of each tracked person.

Tracking trails are created by storing recent center points for each tracking ID.

They help visualize:

* Movement direction

* Walking paths

* Entry and exit behavior

* Tracking consistency

* Possible ID switches

Tracking trails can be enabled or disabled from the Streamlit sidebar.

### 11. Processing Statistics

After video processing, the application displays several statistics.

#### Peak ROI Count

The maximum number of people detected inside the ROI at one time.

#### Unique Tracked IDs

The total number of distinct tracking IDs observed during processing.

#### Entries

The number of recorded `ENTRY` events.

#### Exits

The number of recorded `EXIT` events.

#### Average Dwell Time

The average time people remained inside the ROI.

#### Processing FPS

The approximate number of frames processed per second by the application.

#### Frames Processed

The total number of frames read from the input video.

#### Sampled Frames

The number of frames actually passed through the detection and tracking model after frame skipping.

#### Average ROI Count

The average number of people detected inside the ROI across sampled frames.

## 🧠 System Workflow

The complete processing pipeline follows this sequence:

```
Input Image or Video
        │
        ▼
Upload through Streamlit
        │
        ▼
Read configuration settings
        │
        ├── Confidence threshold
        ├── IoU threshold
        ├── Inference image size
        ├── Frame skip
        └── Tracking algorithm
        │
        ▼
YOLOv8 Person Detection
        │
        ▼
Object Tracking
        │
        ▼
Calculate Bounding-Box Centers
        │
        ▼
Check ROI Membership
        │
        ▼
Compare Current and Previous ROI State
        │
        ├── Outside → Inside = ENTRY
        └── Inside → Outside = EXIT
        │
        ▼
Calculate Dwell Time
        │
        ▼
Draw Bounding Boxes, IDs, ROI, and Trails
        │
        ▼
Generate Processed Video
        │
        ▼
Generate CSV Event Report
        │
        ▼
Display Metrics and Download Results
```

## 📁 Project Structure

```
Day-39/
│
├── app.py
│   └── Streamlit user interface and application controls
│
├── security_monitoring.py
│   └── Detection, tracking, ROI processing, event logging,
│       dwell-time calculation, video generation, and statistics
│
├── requirements.txt
│   └── Python dependencies required to run the project
│
├── README.md
│   └── Project documentation and usage instructions
│
├── test_day39.py
│   └── Basic automated tests for ROI logic and event data types
│
├── .gitignore
│   └── Files and folders excluded from Git tracking
│
├── .streamlit/
│   └── config.toml
│       └── Streamlit theme and upload configuration
│
├── sample_videos/
│   ├── README.md
│   ├── crossing_case_01.mp4
│   └── crossing_case_02.mp4
│
├── outputs/
│   └── Generated processed videos and CSV reports
│
└── screenshots/
    └── Screenshots and demonstration evidence
```

## 🗂️ File Descriptions

### `app.py`

This file contains the Streamlit interface.

Responsibilities include:

* Configuring the Streamlit page

* Displaying sidebar controls

* Uploading images

* Uploading videos

* Calling image-processing functions

* Calling video-processing functions

* Displaying processed results

* Showing processing statistics

* Providing download buttons

* Displaying errors and progress information

The interface is divided into two tabs:

1. Image Monitoring

2. Video Monitoring

### `security_monitoring.py`

This is the main processing module.

It contains the core computer-vision pipeline, including:

* YOLO model loading

* ROI membership checks

* Event dataframe normalization

* ROI drawing

* Bounding-box labels

* Image detection

* Video tracking

* Entry and exit detection

* Dwell-time calculation

* Tracking trails

* Processed-video creation

* CSV report generation

* Performance statistics

Separating this logic from `app.py` makes the project easier to maintain, test, and extend.

### `requirements.txt`

This file lists the required Python packages.

Main dependencies include:

* `streamlit`

* `ultralytics`

* `opencv-python`

* `numpy`

* `pandas`

* `Pillow`

* `lap`

The `lap` package is included because Ultralytics tracking may require it for tracker-related operations.

### `test_day39.py`

This file contains basic tests for important utility functions.

The tests verify:

* ROI boundary behavior

* Points outside the ROI

* Behavior when no ROI is provided

* Event dataframe column types

* Empty event dataframe behavior

These tests help detect regressions when the project is modified.

### `.streamlit/config.toml`

This file configures Streamlit.

It currently:

* Enables a dark theme

* Allows larger uploaded files

The upload limit is configured as:

TOML

```
[server]
maxUploadSize = 500
```

The value is measured in megabytes.

### `sample_videos/`

This folder contains test videos.

For the Day 39 crossing-case evaluation, add:

```
crossing_case_01.mp4
crossing_case_02.mp4
```

The videos should contain people crossing paths or partially occluding one another.

These videos are useful for evaluating:

* Tracking-ID consistency

* ID switches

* Duplicate events

* ROI count stability

* Tracker performance

* The effect of confidence and IoU settings

### `outputs/`

This folder stores generated files such as:

```
day39_processed_video.mp4
day39_event_report.csv
```

Generated outputs should generally not be committed to GitHub unless they are small and specifically required as project evidence.

### `screenshots/`

This folder stores visual evidence for the project.

Recommended screenshots include:

1. Streamlit upload interface

2. Sidebar optimization settings

3. Processed video with tracking IDs

4. ROI overlay

5. Statistics dashboard

6. CSV report download section

7. Processed image output

## 🛠️ Installation

### Step 1: Open the Project Folder

Open PowerShell and navigate to the project:

PowerShell

```
cd "C:\path\to\Day-39"
```

Replace the path with the actual location of your project folder.

### Step 2: Create a Virtual Environment

Using Python 3.13:

PowerShell

```
py -3.13 -m venv .venv
```

### Step 3: Activate the Virtual Environment

PowerShell

```
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run PowerShell with the appropriate local execution-policy configuration or activate the environment from Command Prompt instead.

### Step 4: Upgrade pip

PowerShell

```
python -m pip install --upgrade pip
```

### Step 5: Install Dependencies

PowerShell

```
pip install -r requirements.txt
```

The YOLO model file may be downloaded automatically by Ultralytics during the first inference run.

## ▶️ Running the Application

Start the Streamlit application:

PowerShell

```
streamlit run app.py
```

Streamlit normally opens the application at:

```
http://localhost:8501
```

If port `8501` is already in use, run:

PowerShell

```
streamlit run app.py --server.port 8502
```

## 🌐 Creating a Temporary Public URL

If you need to demonstrate the application remotely, install and configure ngrok.

Then run:

PowerShell

```
ngrok http 8501
```

If the Streamlit application is running on port `8502`, use:

PowerShell

```
ngrok http 8502
```

The generated ngrok URL can be used for temporary demonstrations.

> Important: A public tunnel should only be used for temporary demonstrations. Do not upload private or sensitive video footage.

## 🧪 Running Tests

Run the test file from the project directory:

PowerShell

```
python test_day39.py
```

Expected output:

```
Day 39 tests passed successfully.
```

The tests are designed to validate the supporting logic without requiring a complete video-processing run.

## 📤 Using the Application

### Image Workflow

1. Open the Image Monitoring tab.

2. Upload a JPG, JPEG, or PNG image.

3. Review the original image.

4. Click Run image monitoring.

5. Wait for YOLO inference to complete.

6. Review the processed image.

7. Check the number of detected people.

8. Download the processed image.

### Video Workflow

1. Open the Video Monitoring tab.

2. Upload an MP4, MOV, AVI, or MKV file.

3. Review the input video.

4. Configure the processing settings from the sidebar.

5. Select the tracking algorithm.

6. Choose the inference image size.

7. Set the confidence and IoU thresholds.

8. Choose the frame-skip value.

9. Enable or disable tracking trails.

10. Click Run optimized video monitoring.

11. Wait for processing to finish.

12. Review the processed video.

13. Review the statistics.

14. Download the processed MP4 file.

15. Download the CSV event report.

## ⚙️ Recommended Settings

### Accuracy-Oriented Settings

Use these settings when tracking quality is more important than speed:

```
Confidence: 0.35 – 0.45
IoU: 0.50
Image size: 640
Frame skip: 1
Tracker: ByteTrack or BoT-SORT
Tracking trails: Enabled
```

### Speed-Oriented Settings

Use these settings for longer videos or systems with limited processing power:

```
Confidence: 0.45 – 0.60
IoU: 0.50
Image size: 320 or 416
Frame skip: 2 or 3
Tracker: ByteTrack
Tracking trails: Disabled
```

### Balanced Settings

A practical starting configuration is:

```
Confidence: 0.45
IoU: 0.50
Image size: 512
Frame skip: 1
Tracker: ByteTrack
Tracking trails: Enabled
```

## 📄 CSV Event Report

The generated CSV report contains event-level information.

Example structure:

|
Column

|

Description

|
| --- | --- |
|

`track_id`

|

Unique tracking ID assigned to a person

|
|

`event`

|

`ENTRY` or `EXIT`

|
|

`roi`

|

Name of the monitored region

|
|

`frame`

|

Frame where the event was detected

|
|

`timestamp_s`

|

Event timestamp in seconds

|
|

`entry_time_s`

|

Time when the person entered

|
|

`exit_time_s`

|

Time when the person exited

|
|

`duration_s`

|

Time spent inside the ROI

|
|

`status`

|

Event or session status

|

Example:

```
track_id,event,roi,frame,timestamp_s,entry_time_s,exit_time_s,duration_s,status
7,ENTRY,ROI,150,6.00,6.00,,,active
7,EXIT,ROI,310,12.40,6.00,12.40,6.40,completed
```

The CSV can be opened using:

* Microsoft Excel

* Google Sheets

* Pandas

* LibreOffice Calc

* Any standard text editor

## 📊 Example Statistics

A completed video run may produce statistics similar to:

```
Peak people in ROI: 5
Unique tracked IDs: 12
Entries: 12
Exits: 12
Average dwell time: 8.42 seconds
Processing FPS: 6.75
Frames processed: 900
Sampled frames: 900
Average ROI count: 2.31
```

These values depend entirely on the uploaded video, hardware, model settings, and tracking behavior.

## 🔍 Crossing-Case Evaluation

A major Day 39 improvement is the evaluation of tracking consistency when people cross paths.

For this evaluation:

1. Add two real videos to `sample_videos/`.

2. Use the same settings for both videos.

3. Run the first video using ByteTrack.

4. Record the tracking behavior.

5. Run the second video using ByteTrack.

6. Repeat both tests using BoT-SORT.

7. Compare the results.

Evaluate the following:

* Do tracking IDs remain stable?

* Are IDs switched when people cross?

* Are duplicate `ENTRY` events generated?

* Are `EXIT` events generated correctly?

* Does the ROI count fluctuate unexpectedly?

* Does one tracker perform better than the other?

* Does changing the confidence threshold improve detection?

* Does reducing frame skipping improve identity consistency?

A simple evaluation table can be used:

|
Test Video

|

Tracker

|

Frame Skip

|

ID Stability

|

Duplicate Events

|

Overall Result

|
| --- | --- | --- | --- | --- | --- |
|

Crossing Case 1

|

ByteTrack

|

1

|

Good/Fair/Poor

|

Yes/No

|

Notes

|
|

Crossing Case 1

|

BoT-SORT

|

1

|

Good/Fair/Poor

|

Yes/No

|

Notes

|
|

Crossing Case 2

|

ByteTrack

|

1

|

Good/Fair/Poor

|

Yes/No

|

Notes

|
|

Crossing Case 2

|

BoT-SORT

|

1

|

Good/Fair/Poor

|

Yes/No

|

Notes

|

## 🔧 Optimization Summary

The main optimization techniques used in this project are:

### Configurable Inference

The user can change confidence, IoU, and image size without modifying the source code.

### Frame Skipping

The application can process fewer frames to reduce computation time.

### Model Selection

YOLOv8n is used because it is lightweight and suitable for experimentation and demonstrations.

### Tracker Selection

The user can compare ByteTrack and BoT-SORT depending on the video scenario.

### Person-Only Inference

The pipeline restricts inference to the person class, reducing unnecessary detections.

### Modular Design

The Streamlit interface is separated from the processing logic.

### Data-Type Normalization

Event dataframe columns are explicitly converted to suitable data types. This helps prevent mixed-type dataframe errors during display or CSV generation.

### Efficient Trail Storage

Tracking trails use bounded deques so that the application does not store unlimited historical points.

### Progress Feedback

The application reports progress while processing videos.

### Error Handling

The application handles common failures, including:

* Invalid image uploads

* Unsupported video files

* Unreadable videos

* Output-video creation failures

* Inference errors

* Missing or invalid processing inputs

## ⚠️ Limitations

This project is intended for educational, experimental, and demonstration purposes.

Known limitations include:

1. Tracking IDs may change during severe occlusion.

2. People crossing directly in front of one another may cause ID switches.

3. Poor lighting can reduce detection quality.

4. Very small people may be missed.

5. Frame skipping may reduce event-timing precision.

6. The current ROI implementation uses a rectangular region.

7. The system does not identify people by name.

8. The system does not perform facial recognition.

9. The system does not determine intent or suspicious behavior.

10. The system should not be treated as a fully autonomous security decision-maker.

11. Performance depends on the CPU, GPU, video resolution, and selected inference settings.

12. The generated MP4 codec may vary depending on the operating system and OpenCV installation.

## 🔐 Privacy and Responsible Use

This project processes visual data that may contain people.

When using real-world footage:

* Obtain appropriate permission.

* Avoid uploading sensitive footage to public services.

* Do not use the system for unauthorized surveillance.

* Do not attempt to identify people by name.

* Store generated reports securely.

* Remove private footage before publishing the repository.

* Use anonymized or consented footage for demonstrations whenever possible.

The system detects and tracks visible people; it does not establish identity, intent, or guilt.

## 📦 GitHub Submission Checklist

Before pushing the project to GitHub, verify the following:

* `app.py` is included.

* `security_monitoring.py` is included.

* `requirements.txt` is included.

* `README.md` is complete.

* `test_day39.py` is included.

* `.gitignore` is included.

* `.streamlit/config.toml` is included.

* `sample_videos/README.md` is included.

* Two crossing-case videos were tested.

* Screenshots were added.

* The application runs successfully.

* The test file passes.

* No private videos are committed.

* No large model files are committed.

* No temporary virtual-environment files are committed.

* The GitHub repository link is added to the final submission.

* A demo video is recorded.

* A Streamlit or Hugging Face deployment link is added if available.

## 📝 Suggested Git Commands

From the parent directory containing `Day-39`:

PowerShell

```
git add Day-39
```

Commit the project:

PowerShell

```
git commit -m "Add Day 39 optimized security monitoring"
```

Push the changes:

PowerShell

```
git push origin main
```

If your current branch is not `main`, check it first:

PowerShell

```
git branch
```

## 🎥 Suggested Demo Video Structure

For a 3–5 minute demonstration, use the following structure:

### 1. Introduction — 20–30 seconds

Explain:

* The purpose of the project

* How it improves Day 38

* The main technologies used

### 2. Project Structure — 20–30 seconds

Briefly show:

* `app.py`

* `security_monitoring.py`

* `requirements.txt`

* `test_day39.py`

* `sample_videos/`

* `outputs/`

### 3. Configuration Controls — 30–45 seconds

Demonstrate:

* Confidence slider

* IoU slider

* Image-size selector

* Frame-skip control

* Tracker selection

* Tracking trails

### 4. Image Processing — 20–30 seconds

Show:

* Image upload

* Person detection

* Processed image

* Download button

### 5. Video Processing — 60–90 seconds

Show:

* Video upload

* ROI monitoring

* Tracking IDs

* Tracking trails

* Active ROI count

* Progress indicator

* Processed video output

### 6. Analytics and CSV — 30–45 seconds

Show:

* Peak ROI count

* Unique IDs

* Entry count

* Exit count

* Average dwell time

* CSV report

### 7. Crossing-Case Evaluation — 30–45 seconds

Explain:

* The purpose of the two crossing-case videos

* The comparison between ByteTrack and BoT-SORT

* Any observed ID switches or improvements

### 8. Conclusion — 15–20 seconds

Summarize:

* Performance improvements

* Better configurability

* Event-based reporting

* Future improvement possibilities

## 🔮 Future Improvements

Potential future improvements include:

* Polygon-based ROI selection

* Interactive ROI drawing directly on the first video frame

* Multiple ROIs

* Line-crossing detection

* Restricted-zone alerts

* Email or webhook notifications

* Real-time camera support

* GPU/CPU performance selection

* Automatic tracker-quality evaluation

* ID-switch counting

* Event deduplication using temporal debouncing

* Heatmap generation

* Occupancy-over-time charts

* SQLite or PostgreSQL event storage

* Multi-camera monitoring

* Docker deployment

* Cloud deployment

* Automatic PDF report generation

* Improved handling of occlusion and crowded scenes

* Custom-trained models for specific environments

## 🏁 Conclusion

Day 39 transforms the Day 38 security-monitoring prototype into a more configurable and evaluation-oriented computer-vision application.

The updated system combines:

* YOLOv8 person detection

* Persistent object tracking

* ROI-based monitoring

* Entry and exit event detection

* Dwell-time analysis

* Tracking trails

* Adjustable inference parameters

* Frame-skip optimization

* CSV event reporting

* Processed-video generation

* Streamlit-based interaction

* Automated utility tests

The project demonstrates how a basic detection pipeline can be developed into a more structured security-monitoring system with better performance controls, clearer analytics, and more useful outputs.
