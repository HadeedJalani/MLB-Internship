# Day 18 — Edge Detection & Morphological Operations 🚀

![Python](https://img.shields.io/badge/Python-3.x-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![Streamlit](https://img.shields.io/badge/Deployment-Streamlit-red)

Part of the **MLBench Summer Internship Program**.

This module focuses on **Edge Detection, Morphological Operations, Contour Detection, and Document Boundary Detection using OpenCV**.

The project also includes an interactive Streamlit application called **Document Vision Lab**, which allows users to upload document images, experiment with different computer vision techniques, detect document boundaries, and perform perspective correction.

---

# 📌 Overview

| Task | Edge Detection & Morphological Operations |
|---|---|
| Framework | OpenCV |
| Language | Python |
| Input | Document Images |
| Edge Methods | Sobel, Laplacian, Canny |
| Morphology | Erosion, Dilation, Opening, Closing, Gradient, Top Hat, Black Hat |
| Detection | Contour-based document boundary detection |
| Additional Feature | Perspective correction |
| Interface | Streamlit |
| Deployment | Streamlit Cloud |

---

# 🧠 Concepts Covered

## Edge Detection

Edge detection identifies areas where image intensity changes significantly.

Edges are useful for identifying:

- Object boundaries
- Document borders
- Shapes
- Text regions
- Structural features

This project implements three major edge detection techniques:

- Sobel
- Laplacian
- Canny

---

# 1️⃣ Sobel Operator

The Sobel operator calculates image intensity changes in the horizontal and vertical directions.

OpenCV implementation:

```python
cv2.Sobel()
````

Sobel is useful when directional information is important.

For example:

* Horizontal edges
* Vertical edges
* Shape analysis

---

# 2️⃣ Laplacian Operator

The Laplacian operator uses the second derivative to detect rapid changes in image intensity.

OpenCV implementation:

```python
cv2.Laplacian()
```

It can detect edges in multiple directions but may also respond strongly to noise.

---

# 3️⃣ Canny Edge Detection

Canny is a multi-stage edge detection algorithm.

The general pipeline is:

```text
Grayscale
     ↓
Gaussian Blur
     ↓
Gradient Calculation
     ↓
Non-Maximum Suppression
     ↓
Double Threshold
     ↓
Edge Tracking
```

OpenCV implementation:

```python
cv2.Canny()
```

Canny generally produces cleaner and thinner edges than simple gradient-based methods.

---

# 🔬 Edge Detection Comparison

The application compares:

| Method    | Main Characteristic                     |
| --------- | --------------------------------------- |
| Sobel     | Detects directional gradients           |
| Laplacian | Detects rapid intensity changes         |
| Canny     | Produces refined and well-defined edges |

For document boundary detection, Canny generally provided the most useful result because document borders can be extracted more clearly after threshold tuning and noise reduction.

---

# 🧱 Morphological Operations

Morphological operations modify the structure of objects in binary or grayscale images.

They are particularly useful for:

* Removing noise
* Connecting broken edges
* Filling small gaps
* Extracting image structures
* Improving contour detection

This project implements seven morphological operations.

---

# 1️⃣ Erosion

Erosion removes pixels around object boundaries.

It can be used to:

* Remove small white noise
* Reduce object size
* Separate connected objects

OpenCV:

```python
cv2.morphologyEx(
    image,
    cv2.MORPH_ERODE,
    kernel
)
```

---

# 2️⃣ Dilation

Dilation expands white regions.

It can be used to:

* Connect broken edges
* Fill small gaps
* Increase object thickness

---

# 3️⃣ Opening

Opening is:

```text
Erosion → Dilation
```

It is useful for removing small isolated noise while preserving larger structures.

OpenCV:

```python
cv2.MORPH_OPEN
```

---

# 4️⃣ Closing

Closing is:

```text
Dilation → Erosion
```

It is useful for:

* Closing small holes
* Connecting broken edges
* Creating continuous document boundaries

OpenCV:

```python
cv2.MORPH_CLOSE
```

For this project, closing is particularly useful before contour detection.

---

# 5️⃣ Morphological Gradient

Morphological gradient highlights object boundaries.

It is calculated using:

```text
Dilation − Erosion
```

OpenCV:

```python
cv2.MORPH_GRADIENT
```

---

# 6️⃣ Top Hat

Top Hat extracts small bright regions from an image.

OpenCV:

```python
cv2.MORPH_TOPHAT
```

It can be useful for detecting bright details against a darker background.

---

# 7️⃣ Black Hat

Black Hat extracts small dark regions.

OpenCV:

```python
cv2.MORPH_BLACKHAT
```

It can be useful for identifying dark text or structures on a lighter background.

---

# 🔎 Document Boundary Detection

The main objective of the mini project is to detect the boundary of a document.

The pipeline is:

```text
Input Image
     ↓
Grayscale
     ↓
Gaussian Blur
     ↓
Edge Detection
     ↓
Morphological Processing
     ↓
Contour Detection
     ↓
Contour Sorting
     ↓
Four-Corner Approximation
     ↓
Document Boundary
     ↓
Perspective Correction
```

The application searches for the largest suitable contour that can be approximated as a four-sided polygon.

The detected boundary is then drawn over the original image.

---

# 📐 Perspective Correction

Once four document corners are detected, the application can transform the document into a rectangular view.

The process uses:

```python
cv2.getPerspectiveTransform()
```

and:

```python
cv2.warpPerspective()
```

This is especially useful for documents photographed using a mobile phone at an angle.

Example:

```text
Tilted Document
       ↓
Boundary Detection
       ↓
Four Corner Points
       ↓
Perspective Transformation
       ↓
Straightened Document
```

---

# 🌐 Streamlit Application

The project includes an interactive Streamlit application:

```text
streamlit_app.py
```

The application is called:

## Document Vision Lab

Users can upload a document image and experiment with the complete computer vision pipeline.

---

# 🚀 Application Features

## Image Upload

Users can upload:

* JPG
* JPEG
* PNG

document images.

---

## Processing Presets

The application includes predefined processing modes:

* Clean Scan
* Mobile Photo
* Shadow / Uneven Lighting
* Noisy Document
* Custom

These presets provide different processing parameters for different image conditions.

---

## Manual Controls

The Custom mode allows users to control:

* Edge detection method
* Canny lower threshold
* Canny upper threshold
* Morphological operation
* Kernel size
* Number of iterations

---

## Edge Detection Comparison

The application can display:

* Sobel
* Laplacian
* Canny

side-by-side for direct comparison.

---

## Boundary Detection

The application attempts to detect the document boundary and displays:

* Boundary polygon
* Four corner points
* Contour area
* Document coverage
* Rectangularity
* Detection confidence

---

## Perspective Correction

When a suitable four-sided document boundary is detected, the application can automatically straighten the document.

---

## Processing Statistics

The application reports:

* Original image resolution
* Processing time
* Detected contour area
* Image coverage
* Number of detected corners
* Rectangularity
* Detection confidence

---

## Download Center

Users can download:

* Edge detection result
* Morphological result
* Document boundary result
* Perspective-corrected document

---

# 📂 Project Structure

```text
Day18/
│
├── streamlit_app.py
│
├── opencv_transformations.py
├── transformation_operations.py
│
├── image_enhancement.py
├── enhancement_operations.py
│
├── document_enhancement_tool.py
│
├── requirements.txt
│
├── README.md
│
├── input_images/
│
└── output_images/
    │
    ├── edges/
    ├── morphology/
    ├── boundaries/
    └── comparisons/
```

Input and generated output images are excluded from GitHub using `.gitignore` to keep the repository lightweight.

---

# ⚙️ Installation

Navigate to the Day18 directory:

```bash
cd Day18
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Edge Detection

```bash
python opencv_transformations.py
```

---

# ▶️ Run Morphological Operations

```bash
python enhancement_operations.py
```

---

# ▶️ Run Document Boundary Detection

```bash
python document_enhancement_tool.py
```

---

# 🌐 Run Streamlit Application

```bash
streamlit run streamlit_app.py
```

The application will open locally in the browser.

---

# 📊 Challenge Task

The project was designed to process multiple document images under different conditions.

The test images should include:

* Straight scanned documents
* Mobile phone photographs
* Tilted documents
* Documents with shadows
* Uneven lighting
* Slightly blurred documents

For each document, the processing pipeline can generate:

```text
Original Image
       ↓
Edge Detection
       ↓
Morphological Result
       ↓
Detected Boundary
       ↓
Perspective Corrected Image
```

---

# 🔬 Observations

During experimentation, the quality of document boundary detection depended strongly on:

* Image lighting
* Background complexity
* Document contrast
* Edge strength
* Canny threshold values
* Morphological kernel size
* Morphological operation
* Document orientation

Canny combined with Gaussian Blur and morphological closing generally provided a strong starting point for document boundary detection.

However, no single configuration works perfectly for every document image.

This is why the Streamlit application provides both presets and manual parameter controls.

---

# ⚠️ Challenges

Some of the main challenges encountered were:

### Weak Document Edges

Documents with low contrast against their background sometimes produced weak edges.

### Shadows

Shadows around documents can create additional contours and interfere with boundary detection.

### Uneven Lighting

Different brightness levels across a document can make edge detection inconsistent.

### Tilted Documents

Highly tilted documents can produce irregular contours that are difficult to approximate as four points.

### Background Objects

Objects surrounding a document may produce larger contours than the document itself.

---

# 💡 Best Performing Pipeline

The most useful general-purpose pipeline was:

```text
Grayscale
     ↓
Gaussian Blur
     ↓
Canny Edge Detection
     ↓
Morphological Closing
     ↓
Contour Detection
     ↓
Four-Corner Approximation
     ↓
Perspective Correction
```

The exact thresholds and kernel size may need to be adjusted depending on the input image.

---

# 📚 Learning Outcomes

Through this project, I learned:

* Fundamentals of edge detection
* Sobel edge detection
* Laplacian edge detection
* Canny edge detection
* Canny threshold selection
* Gaussian preprocessing
* Morphological image processing
* Erosion and dilation
* Opening and closing
* Morphological gradient
* Top Hat and Black Hat operations
* Contour detection
* Polygon approximation
* Document boundary detection
* Perspective correction
* OpenCV image preprocessing
* Building interactive Computer Vision applications
* Deploying OpenCV applications using Streamlit

---

# 🔗 Deployment

The Streamlit application is deployed publicly and can be used to upload document images and experiment with the processing pipeline.

**Streamlit App:**

[https://mlb-internship-krc6mpncp3ho7kgjlqgtav.streamlit.app/](https://mlb-internship-krc6mpncp3ho7kgjlqgtav.streamlit.app/)

**GitHub Repository:**

[https://github.com/HadeedJalani/MLB-Internship](https://github.com/HadeedJalani/MLB-Internship)

---

# 👨‍💻 Author

**Hadeed Jalani**

BS Computer Science
MLBench Summer Internship Program

---

# 🏁 Conclusion

Day 18 builds on the image processing concepts learned during the previous sessions.

By combining edge detection, morphological processing, contour analysis, and perspective transformation, a basic document-scanning pipeline can be created using OpenCV.

These techniques provide a strong foundation for more advanced Computer Vision applications such as:

* OCR
* Document scanning
* Object detection
* Image segmentation
* Shape detection
* Automated document analysis

````
