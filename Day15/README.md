# Day 16 — OpenCV Fundamentals & Image Processing Toolkit 🖼️

![Python](https://img.shields.io/badge/Python-3.x-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green)
![Streamlit](https://img.shields.io/badge/Deployment-Streamlit-red)

Part of the **MLBench Summer Internship Program**.

This project focuses on the fundamentals of **OpenCV**, learning how digital images are represented and manipulated using common image processing techniques. It also includes an interactive **Image Processing Toolkit** built with Streamlit that allows users to upload an image and apply various transformations in real time.

---

# 📌 Overview

| Task | Image Processing Toolkit using OpenCV |
|------|----------------------------------------|
| Library | OpenCV (cv2) |
| Language | Python |
| Framework | Streamlit |
| Input | User Uploaded Images |
| Output | Processed Images |
| Deployment | Streamlit Image Processing Toolkit |

---

# 🧠 Concepts Covered

## What is OpenCV?

OpenCV (Open Source Computer Vision Library) is an open-source library used for computer vision and image processing. It provides efficient functions for reading, modifying, analyzing, and saving images and videos.

Common applications include:

- Image Processing
- Face Detection
- Medical Imaging
- Robotics
- Autonomous Vehicles
- Industrial Automation

---

## BGR vs RGB

| BGR | RGB |
|------|------|
| Default format in OpenCV | Standard format for display |
| Blue → Green → Red | Red → Green → Blue |
| Used internally by OpenCV | Used by Matplotlib and Pillow |

OpenCV loads images in **BGR** format, so converting them to **RGB** is necessary before displaying them using visualization libraries.

---

## What are Grayscale Images?

A grayscale image contains only one intensity channel instead of three color channels.

Pixel values range from:

- **0** → Black
- **255** → White

Grayscale images reduce computational complexity and are widely used in feature extraction, edge detection, thresholding, and image analysis.

---

# 📂 OpenCV Practice Programs

## 1. OpenCV Fundamentals

**File:** `opencv_fundamentals.py`

Features:

- Read images
- Display dimensions
- Display channels
- Display file size
- Convert BGR to RGB
- Convert to grayscale
- Save processed images

---

## 2. Basic Image Operations

**File:** `basic_image_operations.py`

Implemented operations:

- Resize
- Crop
- Rotate
- Flip
- Save processed outputs

---

## 3. Drawing Shapes

**File:** `drawing_shapes.py`

Implemented:

- Rectangle
- Circle
- Line
- Polygon
- Custom Text (Name & Date)

---

# 🛠️ Mini Project — Image Processing Toolkit

**File:** `image_processing_toolkit.py`

The toolkit combines all OpenCV operations into a single reusable application.

Features:

- Upload image
- Convert to grayscale
- Resize image
- Rotate image
- Flip image
- Crop image
- Draw shapes
- Add custom text
- Save processed image

---

# 🌐 Streamlit Application

**File:** `streamlit_app.py`

**Live Demo**

https://mlb-internship-iofdnxjvjdatkpvbmofjjd.streamlit.app/

Users can:

- Upload an image
- Resize
- Rotate
- Flip
- Crop
- Convert to grayscale
- Draw shapes interactively
- Select colors
- Adjust thickness
- Undo last drawing
- Download the processed image

---

# 📊 OpenCV Functions Used

- `cv2.imread()`
- `cv2.imwrite()`
- `cv2.cvtColor()`
- `cv2.resize()`
- `cv2.rotate()`
- `cv2.flip()`
- `cv2.rectangle()`
- `cv2.circle()`
- `cv2.line()`
- `cv2.polylines()`
- `cv2.putText()`

---

# 📚 Observations

- OpenCV stores images in BGR format by default.
- RGB conversion is required for correct visualization.
- Grayscale simplifies image processing while preserving important structural information.
- Image resizing changes dimensions without modifying image content.
- Cropping extracts regions of interest.
- Rotation and flipping efficiently change image orientation.
- Interactive drawing tools make image annotation easier.

---

# 🗂️ Project Structure

```text
Day16/
│
├── opencv_fundamentals.py
├── basic_image_operations.py
├── drawing_shapes.py
├── image_processing_toolkit.py
├── streamlit_app.py
├── requirements.txt
├── README.md
├── input_images/
├── output_images/
└── screen_recording.mp4
```

---

# ⚙️ Installation

```bash
pip install -r requirements.txt
```

Run the programs:

```bash
python opencv_fundamentals.py
python basic_image_operations.py
python drawing_shapes.py
python image_processing_toolkit.py
streamlit run streamlit_app.py
```

---

# 🌍 Deployment

**GitHub**

https://github.com/HadeedJalani/MLB-Internship/tree/main/Day16

**Streamlit**

https://mlb-internship-iofdnxjvjdatkpvbmofjjd.streamlit.app/

---

# 📚 Learning Outcomes

Through this project, I learned:

- OpenCV fundamentals
- BGR vs RGB
- Grayscale image processing
- Image resizing, cropping, rotation, and flipping
- Drawing geometric shapes using OpenCV
- Building reusable image processing applications
- Developing interactive Streamlit applications
- Deploying OpenCV projects on Streamlit Cloud

---

# 👨‍💻 Author

**Hadeed Jalani**

BS Computer Science

**MLBench Summer Internship**
