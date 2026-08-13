# Day 19  Contours & Shape Detection with OpenCV

## 📌 Overview

Day 19 of the **MLB Summer Internship** focuses on **Contour Detection, Shape Classification, and Geometric Analysis using OpenCV**.

This project implements a complete classical computer vision pipeline that detects objects in an image, analyzes their contours, calculates geometric measurements, and classifies objects into common geometric shapes.

An interactive **Streamlit application** is also included, allowing users to upload images and perform shape detection without manually running the Python scripts.

---

# 🎯 Project Objectives

The main objectives of this project are to:

* Understand contours in OpenCV
* Detect objects using contours
* Calculate contour area and perimeter
* Draw and visualize object contours
* Calculate bounding rectangles
* Detect and classify geometric shapes
* Measure shape properties
* Filter unwanted contours
* Build an interactive computer vision application
* Export detection results
* Deploy the application using Streamlit

---

# 🔷 Detectable Shapes

The application is designed to detect **10 shape categories**:

|  # | Shape     |
| -: | --------- |
|  1 | Triangle  |
|  2 | Square    |
|  3 | Rectangle |
|  4 | Pentagon  |
|  5 | Hexagon   |
|  6 | Heptagon  |
|  7 | Octagon   |
|  8 | Nonagon   |
|  9 | Circle    |
| 10 | Polygon   |

### Shape Classification

The system primarily uses polygon approximation to determine the number of vertices.

|          Vertices | Classification     |
| ----------------: | ------------------ |
|                 3 | Triangle           |
|                 4 | Square / Rectangle |
|                 5 | Pentagon           |
|                 6 | Hexagon            |
|                 7 | Heptagon           |
|                 8 | Octagon            |
|                 9 | Nonagon            |
| Other / Irregular | Polygon            |

Circles are handled separately using **circularity**, since a circle does not have a fixed number of polygon vertices.

---

# 🔬 Computer Vision Pipeline

The application follows this processing pipeline:

```text
Input Image
     │
     ▼
Grayscale Conversion
     │
     ▼
Gaussian Blur
     │
     ▼
Thresholding
     │
     ▼
Morphological Processing
     │
     ▼
External Contour Detection
     │
     ▼
Contour Filtering
     │
     ▼
Polygon Approximation
     │
     ▼
Shape Classification
     │
     ▼
Geometric Measurements
     │
     ▼
Visualization
     │
     ▼
Export Results
```

---

# 🧠 Core OpenCV Techniques

## 1. Grayscale Conversion

The input image is converted into grayscale to simplify image processing and reduce the number of channels that need to be analyzed.

```python
gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)
```

---

## 2. Gaussian Blur

Gaussian blur is applied before thresholding to reduce small amounts of image noise and produce cleaner contours.

```python
blurred = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)
```

---

## 3. Thresholding

The application supports multiple preprocessing approaches, including:

* Otsu Thresholding
* Adaptive Thresholding
* Binary Thresholding
* Canny-based processing

These options allow the detector to handle images with different lighting, contrast, and background conditions.

---

## 4. Morphological Processing

Optional morphological processing can be applied using:

* Opening
* Closing
* Opening + Closing

These operations help remove small artifacts, fill gaps, and improve object regions before contour detection.

---

# 🔎 Contour Detection

Contours are detected using OpenCV:

```python
contours, _ = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)
```

The project uses:

```text
cv2.RETR_EXTERNAL
```

to focus on the external boundaries of objects and reduce duplicate detections caused by internal contours.

---

# 📐 Contour Measurements

For every detected object, the application calculates several geometric properties.

### Area

```python
cv2.contourArea(contour)
```

The contour area represents the number of pixels enclosed by the detected contour.

### Perimeter

```python
cv2.arcLength(
    contour,
    True
)
```

The perimeter represents the length of the contour boundary.

### Bounding Rectangle

```python
cv2.boundingRect(contour)
```

The bounding rectangle provides:

* X coordinate
* Y coordinate
* Width
* Height

### Circularity

Circularity is calculated using:

```text
Circularity = 4π × Area / Perimeter²
```

A value closer to `1.0` generally indicates a more circular object.

### Solidity

Solidity compares the contour area with the area of its convex hull. It is useful for analyzing how solid or irregular a detected contour is.

### Extent

Extent compares the contour area with the area of its bounding rectangle and provides another useful geometric descriptor.

---

# 🎨 Shape Visualization

The final result displays information such as:

* Object number
* Detected shape
* Confidence score
* Contour boundary
* Bounding rectangle
* Area
* Perimeter

Different shape categories are visually distinguished using different colors.

> **Note:** The confidence score represents the confidence of the geometric classification rules. It is not a machine-learning probability.

---

# 🖥️ Streamlit Application

The project includes an interactive **Streamlit** application that allows users to upload an image and run the complete shape and contour detection pipeline.

## Available Features

### 📤 Image Upload

Users can upload:

* JPG
* JPEG
* PNG
* WEBP

---

### ⚙️ Adjustable Processing

The sidebar provides controls for:

* Threshold method
* Gaussian blur size
* Morphological operation
* Morphology kernel size
* Minimum object area
* Maximum object area ratio

These controls make it possible to tune preprocessing and contour filtering for different input images.

---

### 🔬 Processing Visualization

The application can display:

* Original image
* Grayscale image
* Gaussian-blurred image
* Thresholded image
* Contour detection result
* Final labeled result

---

### 📊 Detection Summary

The application provides counts for all supported shapes:

```text
Triangle
Square
Rectangle
Pentagon
Hexagon
Heptagon
Octagon
Nonagon
Circle
Polygon
```

It also displays the total number of detected objects.

---

### 🔍 Object Analysis

For each detected object, the application provides:

* Object ID
* Shape
* Confidence
* Area
* Perimeter
* Width
* Height
* Circularity
* Solidity
* Number of vertices

---

### 📈 Shape Distribution

A chart displays the number of detected objects for each supported shape.

This makes it easier to analyze images containing multiple types of geometric objects.

---

### 📥 Export

The application supports downloading:

* Binary image
* Contour result
* Final labeled result
* CSV analysis report

The CSV report contains measurements and classification information for each detected object.

---

# 📂 Project Structure

```text
Day19/
│
├── README.md
├── requirements.txt
│
├── contour_detection.py
├── shape_detection.py
├── streamlit_app.py
│
├── input_images/
│
└── output_images/
```

The `input_images/` and `output_images/` folders are intentionally excluded from the GitHub repository.

They are used locally for testing and generating results.

---

# 🚫 GitHub Image Policy

Input and generated output images are **not uploaded to GitHub**.

The repository uses `.gitignore` rules to exclude:

```text
Day19/input_images/
Day19/output_images/
```

This keeps the repository lightweight and avoids committing large image datasets or generated files.

---

# 📦 Installation

Navigate to the Day19 directory:

```bash
cd Day19
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Streamlit Application

Run the following command:

```bash
python -m streamlit run streamlit_app.py
```

The application will start locally and open in a browser.

---

# 🧪 Testing

The system was tested using multiple images containing different geometric shapes.

Testing included:

* Single-shape images
* Multi-shape images
* Different object sizes
* Different object positions
* Different colors
* Different backgrounds
* Multiple shapes in the same image

A dedicated multi-shape test image was also used to verify that individual objects were not incorrectly detected multiple times.

---

# 🏆 Challenge Task

The application was tested on multiple images to compare three main stages:

### 1. Original Image

The original input image before processing.

### 2. Contour Detection Result

The image showing detected external contours.

### 3. Shape Detection Result

The final processed image containing:

* Shape labels
* Bounding boxes
* Measurements
* Confidence scores

The application allows these processed results to be downloaded for further comparison and analysis.

---

# ⚠️ Limitations

This project uses **classical computer vision techniques** rather than a machine-learning model.

Therefore, detection accuracy can depend on:

* Image quality
* Lighting
* Background complexity
* Object overlap
* Shape distortion
* Perspective
* Blur
* Threshold parameters
* Contour quality

The confidence score represents the confidence of the geometric classification rules and **should not be interpreted as a machine-learning probability**.

For best results, images should contain clearly separated geometric objects with reasonably clean and well-defined boundaries.

---

# 💡 Challenges Faced

## Duplicate Object Detection

Initial versions could detect multiple contours belonging to the same object.

This was improved by using:

```python
cv2.RETR_EXTERNAL
```

along with contour filtering.

---

## Noise

Small artifacts could be incorrectly interpreted as objects.

A minimum contour-area threshold was introduced to remove insignificant detections.

---

## Circle Detection

Circles cannot be reliably classified using only polygon vertex count.

Circularity was therefore introduced as an additional geometric feature.

---

## Square vs. Rectangle

Both squares and rectangles contain four vertices.

The system therefore uses bounding-box aspect ratio together with contour geometry to distinguish between the two.

---

## Different Image Conditions

Different images can require different preprocessing parameters.

The Streamlit application therefore exposes processing controls that allow users to tune the detector according to the input image.

---

# 🌍 Real-World Applications

Contour and shape detection can be applied to:

* Industrial inspection
* Manufacturing quality control
* Robotics
* Object measurement
* OCR preprocessing
* Document analysis
* Computer vision
* Image segmentation
* Automated sorting systems
* Shape-based object recognition

---

# 🚀 Future Improvements

Possible future improvements include:

* Perspective correction
* Automatic parameter selection
* More advanced shape descriptors
* Object tracking
* Real-time webcam detection
* Automatic image preprocessing
* Machine-learning-based classification
* YOLO-based object detection
* Improved handling of overlapping objects
* Rotation-invariant shape analysis

---

# 📈 Learning Outcomes

By completing this project, I learned how to:

* Detect contours using OpenCV
* Understand contour hierarchy
* Filter unwanted contours
* Calculate area and perimeter
* Generate bounding rectangles
* Approximate polygons
* Detect geometric shapes
* Calculate circularity
* Calculate solidity
* Analyze object geometry
* Build an interactive Streamlit computer vision application
* Export computer vision results
* Deploy an OpenCV application

---

# 🌐 Live Application

### Streamlit

[Live Streamlit Application](https://mlb-internship-bktjyuknqsnry3gv82rfqj.streamlit.app/?utm_source=chatgpt.com)

The application allows users to upload an image and interactively perform contour detection, shape classification, and geometric analysis.

---

# 👨‍💻 Internship Information

**MLB Summer Internship**

### Day 19

**Project:** Contours & Shape Detection with OpenCV

### Main Technologies

* Python
* OpenCV
* NumPy
* Pandas
* Streamlit
* Pillow

---

# ⭐ Project Summary

This project demonstrates how classical computer vision techniques can be combined into a complete object-analysis pipeline:

```text
Image
  ↓
Preprocessing
  ↓
Thresholding
  ↓
Contour Detection
  ↓
Contour Filtering
  ↓
Shape Classification
  ↓
Geometric Analysis
  ↓
Visualization
  ↓
Export
```

The result is an interactive shape detection system capable of identifying **Triangle, Square, Rectangle, Pentagon, Hexagon, Heptagon, Octagon, Nonagon, Circle, and Polygon** objects from uploaded images.

---

## 👤 Author

**HADEED JALANI**
