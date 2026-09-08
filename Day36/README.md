# Day 36 | AI Traffic Violation Monitoring System


**Author:** Hadeed Jalani

**Livedemo:**
https://mlb-internship-day36.streamlit.app/

An intelligent traffic video analytics system built with **YOLOv8n, ByteTrack, OpenCV, and Streamlit**. The project detects and tracks vehicles in traffic footage, estimates their movement direction, identifies traffic violations, and presents the results through two visually distinct monitoring and analytics interfaces.

This project is part of **Day 36 of the ML Bench Internship**, focusing on practical computer vision, object tracking, rule based event detection, video analytics, and interactive machine learning applications.

---

## Project Overview

Traffic monitoring is an important computer vision application where simply detecting vehicles is not enough. A useful monitoring system must understand where vehicles are moving, maintain their identities across frames, and determine whether their behavior violates predefined traffic rules.

This project combines real time object detection and multi object tracking with a lightweight rule engine.

The system processes traffic videos frame by frame and performs the following operations:

1. Detects vehicles using YOLOv8n
2. Tracks detected vehicles using ByteTrack
3. Assigns persistent tracking IDs
4. Maintains centroid history for every tracked vehicle
5. Estimates vehicle movement direction
6. Compares vehicle movement with the configured road direction
7. Detects sustained wrong way movement
8. Detects vehicle entry into restricted zones
9. Prevents duplicate violation events
10. Generates vehicle and violation statistics
11. Produces processed MP4 output
12. Presents results through an interactive Streamlit application

The system is intentionally designed as an internship and demonstration project rather than a legal enforcement solution.

---

# Key Features

## Vehicle Detection

The system uses **YOLOv8n from Ultralytics** for lightweight vehicle detection.

The detector focuses on the following COCO classes:

| Vehicle Type | COCO Class |
| ------------ | ---------- |
| Car          | car        |
| Motorcycle   | motorcycle |
| Bus          | bus        |
| Truck        | truck      |

Only these vehicle classes are passed to the traffic monitoring logic.

---

## Persistent Vehicle Tracking

Object detection identifies vehicles independently in each frame. Tracking is required to understand whether the vehicle appearing in one frame is the same vehicle appearing in the next frame.

The project uses **ByteTrack** through the Ultralytics tracking interface.

The tracker is called with persistent tracking enabled so that vehicles can maintain consistent IDs while they remain visible.

Conceptually, the processing pipeline is:

```text
Video Frame
     |
     v
YOLOv8n Detection
     |
     v
Vehicle Class Filtering
     |
     v
ByteTrack
     |
     v
Persistent Track ID
     |
     v
Centroid History
     |
     v
Traffic Rule Evaluation
```

This tracking information forms the foundation of the violation detection system.

---

# Direction Estimation

The system estimates vehicle movement using the history of its bounding box centroid.

For every tracked vehicle, the application stores recent centroid positions.

```text
Previous Position
        |
        |       Movement Vector
        |          dx, dy
        v
Current Position
```

The movement vector is calculated from the oldest useful centroid to the newest centroid.

The vector is then normalized to obtain a unit direction vector.

If the vehicle has moved less than the configured minimum pixel distance, its direction is considered unknown.

This prevents small detector fluctuations from being incorrectly interpreted as actual vehicle movement.

---

# Eight Way Traffic Direction

The expected road direction can be configured using eight compass directions.

```text
        Up

Up Left       Up Right

Left              Right

Down Left     Down Right

       Down
```

Supported directions are:

```text
up
down
left
right
up left
up right
down left
down right
```

This allows the system to work with different camera orientations and road layouts.

---

# Wrong Way Detection

Wrong way detection is based on the angle between the expected road direction and the actual vehicle movement direction.

The system uses the dot product between the normalized direction vectors.

```text
angle = arccos(normal road direction · vehicle direction)
```

If the calculated angle exceeds the configured threshold, the vehicle becomes a wrong way candidate.

However, a single incorrect direction estimate does not immediately create a violation.

This is important because real traffic footage contains:

• Detection noise

• Temporary occlusion

• Vehicle stops

• Small centroid movements

• Tracking fluctuations

• Perspective distortion

The system therefore maintains a rolling history of wrong way decisions.

Example:

```text
False
False
True
True
True
True
True
True
True
True
```

After the configured confirmation window is reached, the system checks whether the required percentage of recent frames indicates wrong way movement.

Only then is the violation confirmed.

---

# Wrong Way Event Deduplication

Wrong way violations are modeled as a persistent state.

Once a vehicle has been confirmed as wrong way, its tracking ID is added to a dedicated set.

Conceptually:

```text
wrong_way_ids = {
    vehicle_id_12,
    vehicle_id_27
}
```

This means the same vehicle cannot generate a new wrong way event on every subsequent frame.

Each confirmed wrong way vehicle produces one wrong way event during the current video.

This makes the event statistics meaningful instead of counting the same violation hundreds of times.

---

# Restricted Zone Detection

The second traffic rule identifies vehicles entering a predefined restricted area.

The restricted zone is represented as an axis aligned rectangle.

Instead of storing absolute pixel coordinates, the project stores the zone using fractions of the frame dimensions.

Example:

```json
{
    "x_min": 0.70,
    "x_max": 0.95,
    "y_min": 0.40,
    "y_max": 0.95
}
```

This makes the configuration resolution independent.

The same configuration can therefore be applied to videos with different resolutions.

---

# Restricted Zone Entry Logic

The vehicle centroid is used as the point for the zone test.

A violation occurs only when the vehicle transitions from outside the zone to inside the zone.

```text
Outside
   |
   v
Inside
   |
   v
Record Violation
```

If the vehicle remains inside the zone, another violation is not generated.

If the vehicle leaves the zone and later enters again, another event can be recorded.

This creates a transition based event system rather than a frame based counting system.

---

# False Entry Prevention

The system also handles vehicles that are already inside the restricted zone when tracking begins.

The first observation initializes the vehicle's previous zone state.

It does not immediately generate an entry violation.

For example:

```text
First observation: Inside

Previous state initialized

No violation
```

Only a genuine transition is considered an entry event.

```text
Outside
   |
   v
Inside

Violation recorded
```

This prevents false events at the beginning of a video.

---

# Violation State Management

The central violation state maintains the information required to make decisions across frames.

It keeps track of:

• Vehicle tracking history

• Centroid history

• Direction history

• Wrong way confirmation history

• Confirmed wrong way vehicle IDs

• Previous restricted zone states

• Recorded violation events

• Vehicle types

• Event timestamps

This state based architecture keeps the video processing logic separate from the presentation layer.

---

# Project Architecture

```text
                         Streamlit Application
                                app.py
                                  |
                                  v
                     Traffic Processing Engine
                       traffic_violation.py
                                  |
             +--------------------+--------------------+
             |                    |                    |
             v                    v                    v
         tracker.py       ViolationState         analytics.py
             |                    |                    |
             v                    v                    v
        YOLOv8n             Direction Rules       Dashboard
        ByteTrack            Zone Rules           Statistics
        Track IDs            Event State          Trend Data
             |                    |                    |
             +--------------------+--------------------+
                                  |
                                  v
                         Processed Video Output
                                  |
                                  v
                         Violation Metrics
                         Event Records
```

---

# Project Structure

```text
Day36/
|
├── app.py
├── traffic_violation.py
├── tracker.py
├── analytics.py
├── requirements.txt
├── README.md
|
├── sample_videos/
│   ├── README.md
│   └── traffic_video.mp4
|
├── outputs/
│   ├── variant 1/
│   │   └── .gitkeep
│   └── variant 2/
│       └── .gitkeep
|
├── screenshots/
│   └── README.md
|
└── .streamlit/
    └── config.toml
```

---

# File Responsibilities

## app.py

The Streamlit entry point.

It handles:

• Application layout

• Video upload

• Sample video selection

• Configuration controls

• Variant selection

• Processing controls

• Result display

• Metrics presentation

• Processed video playback

---

## traffic_violation.py

The main video processing and rule engine.

It coordinates:

• Frame processing

• Detection

• Tracking

• Direction estimation

• Wrong way confirmation

• Restricted zone detection

• Event creation

• Output video generation

• Statistics collection

---

## tracker.py

Responsible for object detection and tracking.

It integrates:

• YOLOv8n

• Ultralytics tracking

• ByteTrack

• Vehicle class filtering

• Persistent tracking IDs

• Bounding box information

---

## analytics.py

Responsible for the analytics and dashboard presentation.

It provides:

• Vehicle statistics

• Violation statistics

• Traffic status

• Cumulative violation trends

• Recent violation history

• Vehicle type distribution

• Dashboard components

---

# Output Variants

The project intentionally provides two different visual experiences.

The variants are not simply color changes.

Each variant has a different purpose and composition.

---

# Variant 1 | Wrong Way Detection

Variant 1 is designed as a **movement monitoring interface**.

It focuses on understanding what individual vehicles are doing.

The video displays:

• Vehicle bounding boxes

• Vehicle types

• Persistent tracking IDs

• Centroid trails

• Vehicle direction arrows

• Expected road direction arrow

• Restricted zone overlay

• Wrong way vehicle highlighting

• Live violation counters

The main purpose is visual inspection of vehicle movement.

A typical processing view follows this concept:

```text
Traffic Video

      Vehicle
        ↓
   [ ID 24 ]
      ↘

Expected Direction
        → → → →

Restricted Zone
+----------------+
|                |
|     ZONE       |
|                |
+----------------+

Violations: 3
Vehicles: 18
```

This variant is especially useful for debugging the computer vision pipeline and visually validating the rule engine.

---

# Variant 2 | Traffic Violation Analytics

Variant 2 is designed as an **analytics dashboard**.

Instead of displaying all the visual information from Variant 1, it deliberately simplifies the video and introduces a dashboard focused on statistics.

The composition includes:

• Minimal centroid visualization

• Right side analytics panel

• Total vehicle count

• Total violation count

• Wrong way count

• Restricted zone count

• Car count

• Truck count

• Bus count

• Motorcycle count

• Traffic status classification

• Cumulative violation trend

• Recent violation log

The goal is to transform raw computer vision events into information that is easier to interpret at an operational level.

---

# Traffic Status Classification

The analytics dashboard provides a simplified traffic status based on detected violation activity.

The purpose is not to provide an official traffic classification.

It is a demonstration of how computer vision events can be transformed into higher level analytics.

Possible status categories can include:

```text
Normal
Moderate
High Activity
Critical
```

The exact classification depends on the configured thresholds and observed events.

---

# Vehicle Statistics

Vehicle statistics are calculated using unique tracking IDs.

This is important because the number of detections is not equal to the number of vehicles.

For example, one vehicle visible for 500 frames should still count as:

```text
1 vehicle
```

rather than:

```text
500 detections
```

The system can report:

• Total unique vehicles

• Cars

• Motorcycles

• Buses

• Trucks

---

# Violation Statistics

The system derives violation statistics from the centralized violation state.

The analytics include:

```text
Total Violations
Wrong Way Violations
Restricted Zone Violations
```

Each event can contain information such as:

```text
Vehicle ID
Vehicle Type
Violation Type
Timestamp
```

This makes the resulting data suitable for visualization and further analysis.

---

# Event Representation

A conceptual violation event can be represented as:

```json
{
    "vehicle_id": 24,
    "vehicle_type": "car",
    "violation_type": "wrong_way",
    "timestamp": 18.42
}
```

A restricted zone event can follow the same structure:

```json
{
    "vehicle_id": 31,
    "vehicle_type": "truck",
    "violation_type": "restricted_zone",
    "timestamp": 27.85
}
```

These event objects allow the analytics layer to remain independent from the video rendering logic.

---

# Resolution Independent Zone Configuration

Restricted zones use normalized coordinates.

For example:

```json
{
    "x_min": 0.70,
    "x_max": 0.95,
    "y_min": 0.40,
    "y_max": 0.95
}
```

The values represent fractions of the frame.

For a frame width of `1920` pixels:

```text
x_min = 0.70 × 1920
```

For a frame height of `1080` pixels:

```text
y_min = 0.40 × 1080
```

This approach avoids hardcoding pixel coordinates for a specific video resolution.

---

# Direction ROI

For scenes where multiple traffic directions are visible, the system can use a direction region of interest.

This is particularly useful for divided roads.

For example:

```text
+----------------------------------+
|                                  |
|       Opposite Traffic           |
|                                  |
|----------------------------------|
|                                  |
|       Monitored Traffic          |
|                                  |
+----------------------------------+
```

The wrong way rule can then focus only on the relevant carriageway.

This reduces incorrect classifications when vehicles traveling in another legitimate direction appear in the camera view.

---

# Processing Pipeline

The complete processing flow is:

```text
Input Video
     |
     v
Read Frame
     |
     v
YOLOv8n Detection
     |
     v
Filter Vehicle Classes
     |
     v
ByteTrack Tracking
     |
     v
Assign Persistent IDs
     |
     v
Calculate Centroids
     |
     v
Update Centroid History
     |
     v
Estimate Direction
     |
     +----------------------+
     |                      |
     v                      v
Wrong Way Rule        Restricted Zone Rule
     |                      |
     v                      v
Confirmation           Entry Transition
     |                      |
     +----------+-----------+
                |
                v
        ViolationState
                |
        +-------+-------+
        |               |
        v               v
   Video Rendering   Analytics
        |               |
        v               v
 Processed MP4     Metrics / Events
```

---

# Technology Stack

| Technology  | Purpose                          |
| ----------- | -------------------------------- |
| Python      | Core programming language        |
| YOLOv8n     | Vehicle detection                |
| Ultralytics | Detection and tracking interface |
| ByteTrack   | Multi object tracking            |
| OpenCV      | Video processing and rendering   |
| Streamlit   | Interactive web application      |
| NumPy       | Numerical operations             |
| Pandas      | Analytics and data handling      |

---

# Installation

Clone the repository and move into the project directory.

```bash
git clone YOUR_REPOSITORY_URL

cd Day36
```

Create a virtual environment.

```bash
python -m venv .venv
```

## Windows

```bash
.venv\Scripts\activate
```

## Linux and macOS

```bash
source .venv/bin/activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

# Running the Application

Start Streamlit using:

```bash
streamlit run app.py
```

The application will open in your browser.

You can then:

1. Select a sample traffic video or upload your own video
2. Select the desired monitoring variant
3. Configure the expected traffic direction
4. Configure the restricted zone
5. Configure detection and confirmation parameters
6. Start processing
7. Review the processed video
8. Inspect the generated analytics
9. Download the processed output

---

# YOLO Model

The project uses:

```text
YOLOv8n
```

The nano model is selected because it provides a good balance between detection capability and computational requirements.

The first inference run may automatically download the model through Ultralytics.

No manual model download is required when the environment has internet access and the Ultralytics model loading process is configured normally.

---

# Streamlit Deployment

The project is designed for deployment using **Streamlit Community Cloud**.

Recommended repository structure:

```text
your repository/
|
├── app.py
├── traffic_violation.py
├── tracker.py
├── analytics.py
├── requirements.txt
├── sample_videos/
├── outputs/
└── .streamlit/
    └── config.toml
```

The Streamlit entry point is located at the project root:

```text
app.py
```

This keeps deployment configuration simple.

## Deployment Steps

1. Push the project to GitHub
2. Open Streamlit Community Cloud
3. Create a new application
4. Select the GitHub repository
5. Select the required branch
6. Set the main file to `app.py`
7. Configure Python 3.12 if required
8. Deploy the application
9. Upload or select a traffic video
10. Test both visual variants

---

# Sample Videos

Traffic videos should be stored inside:

```text
sample_videos/
```

A sample video can optionally have a matching configuration file.

Example:

```text
madrid_intersection.mp4
madrid_intersection_config.json
```

The configuration file can contain video specific calibration settings such as:

• Expected traffic direction

• Restricted zone coordinates

• Direction ROI

• Rule thresholds

All sample footage must be legally suitable for redistribution and demonstration.

The project includes:

```text
sample_videos/README.md
```

for documenting the source and licensing information of each sample.

---

# Responsible Data Usage

Traffic footage must be selected carefully.

Only use videos whose licensing terms permit the intended use.

Before adding a video to the repository, document:

```text
Source
License
Original URL
Usage permission
Attribution requirements
```

Do not commit copyrighted traffic footage unless redistribution is explicitly permitted.

---

# Limitations

## Camera Perspective

Direction estimation occurs in image coordinates rather than real world coordinates.

Perspective effects can therefore influence direction estimates.

Sharp turns, lane merges, and unusual camera angles can create incorrect headings.

---

## Divided Roads

When both directions of traffic are visible, vehicles from another carriageway may be interpreted incorrectly if the monitored region is not configured correctly.

A direction ROI can help restrict the analysis to the intended road area.

---

## Small Vehicles

YOLOv8n is a lightweight detection model.

Very distant vehicles may occupy only a small number of pixels and can therefore be difficult to detect reliably.

Performance can also decrease with:

• Night footage

• Heavy blur

• Severe compression

• Strong occlusion

• Poor lighting

• Unusual camera angles

---

## Tracking Identity Switches

ByteTrack provides strong multi object tracking performance, but no tracking system is perfect.

Heavy occlusion or vehicles disappearing for long periods can cause a new tracking ID to be assigned.

This can affect the continuity of vehicle specific violation history.

---

## Camera Movement

The direction estimation system assumes a relatively stable camera.

A moving or heavily shaking camera can cause apparent motion in the scene and reduce the reliability of direction calculations.

---

## Rule Based Detection

The violation system uses configurable geometric and temporal rules.

It does not understand traffic laws in the same way as a human traffic officer.

The output should therefore be treated as a computer vision monitoring result rather than an official legal decision.

---

# Performance Considerations

The application is designed primarily for demonstration and internship purposes.

Processing performance depends on:

• CPU or GPU availability

• Video resolution

• Video frame rate

• Number of vehicles

• Scene complexity

• Detection settings

• Tracking workload

For cloud demonstrations, a configurable frame processing limit can be used to keep inference practical on shared CPU resources.

For large scale real time traffic monitoring, dedicated GPU infrastructure and optimized inference pipelines would be more appropriate.

---

# Testing Checklist

Before submitting or demonstrating the project, verify the following:

```text
[ ] Variant 1 works with at least three traffic videos

[ ] Vehicle IDs remain reasonably stable

[ ] Vehicle classes are displayed correctly

[ ] Normal direction arrow matches the configured direction

[ ] Wrong way confirmation requires multiple frames

[ ] Confirmed wrong way vehicles generate one event

[ ] Restricted zone entry generates an event only on entry

[ ] Vehicles already inside the zone do not create false entry events

[ ] Vehicles leaving and re entering can generate another entry event

[ ] Vehicle statistics use unique tracking IDs

[ ] Variant 2 has a visibly different composition

[ ] Analytics counters update correctly

[ ] Cumulative violation trends are generated

[ ] Recent violation logs contain useful information

[ ] Processed MP4 files play correctly

[ ] Downloaded output videos are valid

[ ] Sample video licensing information is documented

[ ] Streamlit deployment works successfully

[ ] An unseen uploaded video can be processed
```

---

# Challenges Addressed

This project demonstrates several practical computer vision challenges.

### Detection versus Tracking

Detection identifies objects, while tracking maintains their identities across time.

### Noisy Motion Estimation

Centroid histories and minimum movement thresholds reduce false direction estimates.

### Temporal Confirmation

Wrong way detection requires multiple supporting frames rather than trusting one noisy prediction.

### Event Deduplication

Different violation types require different event models.

Wrong way uses a sticky confirmed state.

Restricted zone uses an outside to inside transition.

### Resolution Independence

Normalized zone coordinates allow configurations to work across different video resolutions.

### Analytics Separation

The processing engine produces structured state and events while the analytics layer turns them into a user friendly dashboard.

---

# Future Improvements

Possible future improvements include:

• Perspective aware direction estimation

• Automatic lane detection

• Polygon based restricted zones

• Multiple restricted zones

• Speed estimation

• License plate recognition

• Traffic density estimation

• Vehicle counting across virtual lines

• Automatic camera calibration

• GPU accelerated inference

• Real time camera streams

• Database backed event storage

• Advanced tracking with ReID

• Alert notifications

• Historical traffic analytics

• Heatmap generation

• Automatic report generation

• Multi camera monitoring

These improvements could transform the prototype into a more advanced traffic intelligence platform.

---

# Project Learning Outcomes

This project demonstrates practical experience with:

```text
Computer Vision
Object Detection
Multi Object Tracking
YOLOv8
ByteTrack
OpenCV
Centroid Tracking
Vector Mathematics
Dot Product
Angle Calculation
Rule Based Detection
Temporal Event Confirmation
State Management
Video Processing
Data Analytics
Streamlit
Cloud Deployment
Software Architecture
```

More importantly, the project demonstrates how individual machine learning components can be combined into a complete application rather than being used as isolated models.

---

# Internship Context

**Day 36 of the ML Bench Internship** focuses on building a practical computer vision system that connects machine learning inference with application logic and user facing analytics.

The project goes beyond basic object detection by introducing:

```text
Detection
      +
Tracking
      +
Motion Analysis
      +
Rule Engine
      +
Event Management
      +
Analytics
      +
Interactive Visualization
```

This makes the project a complete end to end machine learning application.

---

# Attribution

### Project

**AI Traffic Violation Monitoring System**

### Internship

**ML Bench Internship**

### Day

**Day 36**

### Author

**Hadeed Jalani**

---

# Disclaimer

This project is intended for educational, research, demonstration, and internship purposes.

It is not an automated legal enforcement system.

Traffic violations detected by this application are based on computer vision estimates and configurable rules. Detection errors, tracking failures, camera perspective, environmental conditions, and other factors can affect results.

No violation generated by this system should be treated as an official legal citation without independent human verification.

---

# Final Summary

The **AI Traffic Violation Monitoring System** combines modern object detection, multi object tracking, motion analysis, geometric reasoning, temporal confirmation, and interactive analytics into a single Streamlit application.

The system detects vehicles using YOLOv8n, tracks them using ByteTrack, maintains persistent tracking IDs, estimates their movement direction from centroid histories, and evaluates two different traffic rules.

The first rule identifies sustained wrong way movement.

The second identifies vehicle entry into restricted zones.

A dedicated state management layer prevents duplicate events and maintains the information required to calculate meaningful vehicle and violation statistics.

The project then presents the results through two deliberately different interfaces.

**Variant 1** focuses on visual traffic monitoring and movement understanding.

**Variant 2** focuses on traffic violation analytics and operational insights.

Together, these components demonstrate a complete computer vision workflow from raw video input to intelligent event detection, structured analytics, and interactive visualization.

**Built as part of Day 36 of the ML Bench Internship by Hadeed Jalani.**
