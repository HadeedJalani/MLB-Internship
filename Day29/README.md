<h1 align="center">🎯 Smart Object Tracking System</h1>
<h3 align="center">Multi-Object Detection & Tracking with YOLO11 + ByteTrack / BoT-SORT</h3>

<p align="center">
  <strong>Upload a video, track every object across frames with a persistent ID, and download the annotated result.</strong>
</p>

<p align="center">
  Built using <strong>Ultralytics YOLO11</strong> for detection, <strong>ByteTrack</strong> and
  <strong>BoT-SORT</strong> for multi-object tracking, <strong>OpenCV</strong> for video processing,
  and deployed through an interactive <strong>Streamlit</strong> application.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/model-YOLO11-orange" />
  <img src="https://img.shields.io/badge/trackers-ByteTrack%20%7C%20BoT--SORT-blue" />
  <img src="https://img.shields.io/badge/deployed%20with-Streamlit-red" />
</p>

---

## 📌 Project Overview

Unlike single-frame object detection, this project performs **multi-object tracking** — every
detected object is assigned a persistent ID that follows it across the entire video, even through
partial occlusion and re-appearance, rather than being re-detected as a new object in every frame.

The complete pipeline:

- 🧠 Frame-by-frame YOLO11 detection
- 🔗 ByteTrack / BoT-SORT tracking association across frames
- 🆔 Persistent unique tracking IDs per object
- 🎨 Per-ID color-coded bounding boxes and labels
- 📊 Live per-class detection and unique-object statistics
- 🎥 Annotated output video generation
- 🌐 Interactive Streamlit deployment with adjustable confidence and tracker choice

---

## 🎯 Objective

Build a video-based tracking system that answers not just *"what objects are in this frame?"*
but *"how many distinct objects appeared across the whole video, and where did each one go?"*

```
Input Video
      │
      ▼
YOLO11 Detection (per frame)
      │
      ▼
Tracker (ByteTrack / BoT-SORT)
      │
      ├── Persistent Track IDs
      ├── Class Labels
      └── Confidence Scores
      │
      ▼
Annotated Frame (color-coded per ID)
      │
      ▼
Output Video + Statistics
```

---

## 🚀 Key Features

| Feature | Description |
|---|---|
| 🧠 **YOLO11 Detection** | Object detection on every frame using Ultralytics YOLO11 |
| 🔗 **Selectable Tracker** | Choose between ByteTrack (fast, lightweight) or BoT-SORT (stronger identity tracking, better occlusion handling) |
| 🆔 **Persistent Tracking IDs** | Each object keeps the same ID across frames instead of being re-detected as new |
| 🎨 **Consistent Per-ID Colors** | Bounding box color is deterministically generated from the track ID, so the same object stays visually identifiable throughout the video |
| ⚙️ **Adjustable Confidence Threshold** | Control the minimum detection confidence from the sidebar |
| 📊 **Live Statistics** | Unique object count, classes detected, per-class detection frequency, and unique tracked objects per class |
| 🎬 **Live Overlay Counter** | Running "Tracked Objects" count burned into the output video itself |
| 💾 **Download Support** | Download the fully annotated tracking video directly from the app |

---

## ⚡ Why Tracking, Not Just Detection?

Plain object detection tells you what's in a single frame. It has no memory — the same physical
object can be re-counted every frame, and there's no way to answer "how many *unique* objects
appeared in this video?"

**Multi-object tracking** solves this by associating detections across frames using motion and
appearance cues, assigning each real-world object one consistent ID for its entire time on screen.

| Tracker | Strengths | Trade-off |
|---|---|---|
| **ByteTrack** | Fast, lightweight, excellent for high-density scenes (e.g. traffic) | Weaker under heavy occlusion |
| **BoT-SORT** | Stronger identity re-association, better occlusion handling | More computationally intensive |

---

## 🤖 Model Used

This project uses the stock pretrained **YOLO11n** (nano) model:

```python
model = YOLO("yolo11n.pt")
```

YOLO11n is the fastest, smallest variant in the YOLO11 family — chosen here for responsive
frame-by-frame inference across arbitrary user-uploaded videos, since this is a general-purpose
tracker rather than a domain-specific detector trained on one dataset.

> **Trade-off:** the nano model prioritizes speed over accuracy. On small, distant, or
> visually ambiguous objects (e.g. birds at a distance vs. kites or aircraft silhouettes), it can
> misclassify between visually similar COCO classes. See [Known Limitations](#-known-limitations--tuning) below.

---

## 📂 Project Structure

```
Day29/
│
├── tracker/
│   └── object_tracker.py     # Core ObjectTracker class — detection + tracking + stats
│
├── app.py                    # Streamlit deployment app
├── requirements.txt
└── README.md
```

---

## ⚙️ How It Works

### `ObjectTracker` (`tracker/object_tracker.py`)

- Loads a YOLO model and maps the selected tracker name to its Ultralytics config
  (`bytetrack.yaml` / `botsort.yaml`)
- `process_video()` reads the input video frame by frame, runs `model.track(..., persist=True)`
  to get detections *and* persistent IDs in one call, draws color-coded boxes + labels, overlays
  a running tracked-object counter, and writes each annotated frame to the output video
- Aggregates statistics as it goes: unique track IDs seen, classes detected, per-class detection
  counts, and unique tracked objects per class
- Returns a single statistics dictionary once the video finishes processing

### `app.py`

- Streamlit UI: tracker selection, confidence slider, video upload, start button
- Renders the input video, runs `ObjectTracker.process_video()` with a live progress callback,
  then displays the input/output videos side by side along with the statistics dashboard
- Includes two compatibility helpers (`compat_widget`, `render_html`) so the app degrades
  gracefully across different installed Streamlit versions instead of hard-erroring

---

## 🌐 Streamlit Application

### Application Features

- 🎥 **Video Upload** — MP4, AVI, MOV, MKV
- ⚙️ **Tracker Selection** — ByteTrack or BoT-SORT, chosen from the sidebar
- 🎚️ **Confidence Slider** — adjustable from 0.10 to 0.90
- 📊 **Live Progress** — real-time frame-processing progress bar during tracking
- 🖼️ **Side-by-Side Comparison** — original vs. tracked output video
- 📈 **Statistics Dashboard** — frames processed, unique objects, classes detected, processing time
- 🆔 **Full Tracking ID List** and **Classes Detected** breakdown
- ⬇️ **Download** — save the annotated tracking video

---

## 🚀 Running Locally

**1. Clone the Repository**

```bash
git clone https://github.com/HadeedJalani/MLB-Internship.git
cd MLB-Internship/Day29
```

**2. Install Dependencies**

```bash
pip install -r requirements.txt
```

**3. Launch Streamlit**

```bash
streamlit run app.py
```

Streamlit will provide a local URL similar to `http://localhost:8501`. `yolo11n.pt` downloads
automatically via Ultralytics on first run if not already cached locally.

---

## 🎛️ Known Limitations & Tuning

Because this app uses the stock, general-purpose `yolo11n.pt` rather than a domain-specific
model, it can confuse visually similar COCO classes on small or distant objects — for example,
a distant flying bird occasionally being classified as `kite` or `airplane`. This is expected
nano-model behavior, not a bug in the tracking pipeline. Two easy levers to improve results:

1. **Raise the confidence threshold** (try 0.55–0.65) to filter out the lower-confidence,
   ambiguous detections that drive most of the misclassification.
2. **Swap to a larger base model** — change `MODEL_PATH = "yolo11n.pt"` to `"yolo11s.pt"` or
   `"yolo11m.pt"` in `app.py` for meaningfully better accuracy, at the cost of slower per-frame
   inference (more noticeable on CPU).

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Ultralytics | YOLO model loading, detection, and built-in tracker integration |
| YOLO11 | Object detection architecture |
| ByteTrack / BoT-SORT | Multi-object tracking algorithms |
| OpenCV | Video I/O, frame annotation, drawing |
| Streamlit | Interactive web application |

---

## 🎓 Key Learnings

Through this project, hands-on experience was gained with:

- The difference between per-frame detection and cross-frame multi-object tracking
- Using Ultralytics' built-in `model.track(persist=True)` API
- Comparing tracker algorithms (ByteTrack vs. BoT-SORT) and their trade-offs
- Deterministic per-ID visualization (consistent color per tracked object)
- Aggregating per-video statistics from per-frame detection streams
- Handling cross-version Streamlit API compatibility issues in deployment
- Debugging a subtle Markdown/HTML rendering issue caused by string indentation

---


## LIVE-DEMO
---

https://mlb-internship-day29.streamlit.app/


## 👨‍💻 Author

**Hadeed Jalani**
Final-Year Computer Science Student, University of Lahore

Focused on: **Artificial Intelligence • Machine Learning • Computer Vision • Full-Stack Development**

<p align="center">
  <strong>Built as part of the MLBench Summer Internship — Custom Object Detection Journey 🚀</strong>
</p>
