
# Computer Vision Image Processing Studio

> A professional interactive computer vision application built with Python, OpenCV, NumPy, PIL, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge\&logo=opencv\&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20Application-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=for-the-badge\&logo=numpy\&logoColor=white)
![PIL](https://img.shields.io/badge/Pillow-Image%20Processing-3776AB?style=for-the-badge\&logo=python\&logoColor=white)

---

## About the Project

**Computer Vision Image Processing Studio** is an interactive web-based image processing application developed as part of the **MLB Summer Internship  Day 21** task.

The application combines multiple computer vision techniques learned throughout the internship into a single, user-friendly interface.

Instead of running separate Python scripts for every image processing operation, users can upload an image, select an operation, configure its parameters interactively, preview the result, and download the processed image.

The application also supports a **processing pipeline mode**, allowing users to combine multiple operations and apply them sequentially to the same image.

The goal of this project is to demonstrate how traditional computer vision techniques can be organized into a reusable application and exposed through a practical web interface.

---

## Live Application

### Streamlit Application

**Live Demo:**
https://mlb-internship-65q6mnpftea7pdvxebbaz9.streamlit.app/

> The application is publicly deployed using Streamlit and can be used directly from a web browser.

### GitHub Repository

**Repository:**
`https://github.com/HadeedJalani/MLB-Internship`

---

# Project Highlights

The application provides:

* Image upload through a web interface
* Real-time processing controls
* Multiple computer vision operations
* Individual operation mode
* Multi-operation pipeline mode
* Adjustable processing parameters
* Original vs processed image comparison
* Interactive image previews
* Shape detection
* Contour detection
* Image enhancement
* Downloadable processed results
* Clean Streamlit-based user interface
* Modular and reusable processing functions

---

# Available Image Processing Operations

The application includes the following computer vision operations.

## 1. Grayscale Conversion

Converts a color image into a grayscale image.

Grayscale reduces an image from three color channels to a single intensity channel, making it useful for many computer vision tasks.

**Technique:**

```python
cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

---

## 2. Gaussian Blur

Applies Gaussian smoothing to reduce image noise and small details.

Gaussian blur is particularly useful as a preprocessing step before edge detection.

The application allows the user to interactively control the blur kernel size.

**Example:**

```python
cv2.GaussianBlur(image, (kernel, kernel), 0)
```

---

## 3. Edge Detection

The application uses Canny Edge Detection to identify strong intensity transitions within an image.

Users can interactively adjust:

* Lower threshold
* Upper threshold

This allows users to observe how different threshold values affect detected edges.

**Example:**

```python
cv2.Canny(image, lower_threshold, upper_threshold)
```

---

## 4. Image Rotation

Allows the image to be rotated according to a selected angle.

Rotation is performed using an affine transformation matrix.

**Example:**

```python
cv2.getRotationMatrix2D(center, angle, scale)
```

---

## 5. Image Enhancement

Improves the visual appearance of an image by adjusting parameters such as:

* Brightness
* Contrast

This makes the application useful for experimenting with different image enhancement settings.

---

## 6. Contour Detection

Contours are used to identify boundaries of objects within an image.

The application detects contours after converting the image into a suitable representation.

**Example:**

```python
cv2.findContours(...)
```

Contours are useful in:

* Object analysis
* Shape detection
* Boundary detection
* Segmentation tasks

---

## 7. Shape Detection

The application detects common geometric shapes using contours and polygon approximation.

The supported detectable shapes include:

| Shape     | Detection |
| --------- | --------- |
| Triangle  | ✓         |
| Square    | ✓         |
| Rectangle | ✓         |
| Pentagon  | ✓         |
| Hexagon   | ✓         |
| Heptagon  | ✓         |
| Octagon   | ✓         |
| Nonagon   | ✓         |
| Circle    | ✓         |
| Polygon   | ✓         |

Polygon approximation is used to estimate the number of vertices in detected contours.

**Example:**

```python
approx = cv2.approxPolyDP(
    contour,
    epsilon,
    True
)
```

The number of detected vertices is then used to classify the shape.

For example:

* 3 vertices → Triangle
* 4 vertices → Square / Rectangle
* 5 vertices → Pentagon
* 6 vertices → Hexagon
* 7 vertices → Heptagon
* 8 vertices → Octagon
* 9 vertices → Nonagon
* Higher vertex counts → Polygon

Circular objects are additionally evaluated using contour characteristics such as circularity.

---

# Interactive Processing Controls

One of the main goals of the application is to make computer vision processing interactive.

Instead of hard-coding parameters, users can adjust processing values directly from the Streamlit sidebar.

Depending on the selected operation, controls may include:

### Blur

* Blur kernel size

### Canny Edge Detection

* Lower threshold
* Upper threshold

### Rotation

* Rotation angle

### Enhancement

* Brightness
* Contrast

### Shape Detection

* Contour filtering
* Detection sensitivity
* Shape approximation settings

This allows users to immediately observe how parameter changes affect the output.

---

# Processing Modes

The application provides two major processing modes.

## Single Operation

In Single Operation mode, the user selects one image processing technique.

**Example:**

```text
Original Image
      ↓
Grayscale
      ↓
Processed Image
```

This mode is useful when experimenting with a specific computer vision technique.

---

## Chain Multiple Filters

The application also supports a processing pipeline.

Users can select multiple operations and apply them sequentially.

For example:

```text
Original Image
      ↓
Brightness & Contrast
      ↓
Gaussian Blur
      ↓
Edge Detection
      ↓
Final Output
```

This provides a more realistic computer vision workflow where multiple preprocessing operations are combined.

The order of operations matters because each operation receives the output of the previous operation.

---

# Example Processing Pipeline

A typical pipeline could be:

```text
Input Image
     │
     ▼
Brightness & Contrast
     │
     ▼
Grayscale
     │
     ▼
Gaussian Blur
     │
     ▼
Canny Edge Detection
     │
     ▼
Processed Image
```

This demonstrates how individual image processing techniques can be combined into a reusable computer vision workflow.

---

# Application Workflow

The application follows a simple workflow:

```text
              Upload Image
                   │
                   ▼
          Select Processing Mode
                   │
             ┌─────┴─────┐
             │           │
             ▼           ▼
       Single Mode    Pipeline Mode
             │           │
             └─────┬─────┘
                   │
                   ▼
          Configure Parameters
                   │
                   ▼
           Process Image
                   │
                   ▼
        Compare Input / Output
                   │
                   ▼
          Download Result
```

---

# Technology Stack

## Programming Language

### Python

Python is used as the primary programming language because of its extensive ecosystem for computer vision, numerical computing, and rapid application development.

---

## Computer Vision

### OpenCV

OpenCV is responsible for the majority of image processing operations, including:

* Color conversion
* Gaussian filtering
* Canny edge detection
* Image transformations
* Contour detection
* Shape detection
* Geometric analysis

---

## Numerical Processing

### NumPy

NumPy is used for efficient numerical operations and image array manipulation.

---

## Image Processing

### Pillow

Pillow is used where PIL-based image handling is useful, particularly for compatibility with Streamlit image uploads and image conversion.

---

## Web Application

### Streamlit

Streamlit provides the interactive web interface.

It allows Python-based computer vision functionality to be exposed through a browser without requiring a separate frontend framework.

---

# Project Structure

```text
Day21/
│
├── app.py
│   └── Main Streamlit application
│
├── requirements.txt
│   └── Python dependencies
│
├── README.md
│   └── Project documentation
│
└── input_images/
    └── Local testing images
```

Generated output files and large image assets are intentionally excluded from version control where appropriate.

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/HadeedJalani/MLB-Internship.git
```

Navigate to the project:

```bash
cd MLB-Internship/Day21
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Requirements

The project uses the following major dependencies:

* `streamlit`
* `opencv-python`
* `numpy`
* `Pillow`

The exact versions are maintained in:

```text
requirements.txt
```

---

# Running the Application Locally

From the `Day21` directory:

```bash
streamlit run app.py
```

Streamlit will start a local development server.

Open the displayed local URL in your browser.

Typically:

```text
http://localhost:8501
```

---

# How to Use the Application

## Step 1 — Upload an Image

Use the image uploader in the sidebar to upload an image.

Supported formats include:

* JPG
* JPEG
* PNG
* BMP
* WEBP

---

## Step 2 — Select Processing Mode

Choose between:

**Single Operation**

or:

**Chain Multiple Filters**

---

## Step 3 — Select an Operation

Choose the desired computer vision operation.

Examples:

* Grayscale
* Blur
* Edge Detection
* Rotation
* Image Enhancement
* Contour Detection
* Shape Detection

---

## Step 4 — Configure Parameters

Adjust the available sliders and controls.

For example, when using Canny Edge Detection:

* Lower Threshold
* Upper Threshold

When using Gaussian Blur:

* Blur Kernel Size

---

## Step 5 — Process the Image

Run the selected operation or pipeline.

The application displays:

* Original image
* Processed image
* Processing information
* Selected parameters

---

## Step 6 — Download the Result

The processed image can be downloaded directly from the application.

This makes it possible to use the generated output in other projects or workflows.

---

# Shape Detection Methodology

Shape detection is implemented using OpenCV contours and polygon approximation.

The general process is:

```text
Input Image
     ↓
Grayscale
     ↓
Noise Reduction
     ↓
Edge / Threshold Processing
     ↓
Contour Extraction
     ↓
Polygon Approximation
     ↓
Vertex Counting
     ↓
Shape Classification
```

The approximation tolerance is important because real images rarely contain perfectly clean geometric boundaries.

The system therefore uses contour filtering and polygon approximation to improve detection stability.

Detected objects can be classified as:

* Triangle
* Square
* Rectangle
* Pentagon
* Hexagon
* Heptagon
* Octagon
* Nonagon
* Circle
* Polygon

---

# Why Shape Detection Can Be Challenging

Real-world images contain:

* Noise
* Shadows
* Reflections
* Uneven lighting
* Perspective distortion
* Blurred boundaries
* Irregular object edges

Because of these factors, shape detection is not simply a matter of counting contour vertices.

The application therefore uses preprocessing and contour-based analysis to improve the reliability of classification.

This also demonstrates an important computer vision concept:

> Good preprocessing often has a major impact on downstream detection performance.

---

# Error Handling

The application includes handling for common image-processing problems, including:

* Invalid image uploads
* Unsupported image formats
* Incorrect image channel counts
* Grayscale images being processed again as color images
* Processing failures
* Invalid parameter combinations

Particular care is taken with image channel handling because OpenCV operations such as:

```python
cv2.cvtColor()
```

require the input image to have the expected number of channels.

The application checks the image representation before applying color conversions where necessary.

---

# Design Principles

The application was designed around several practical software engineering principles.

## Reusability

Processing operations are implemented as reusable functions rather than placing all processing logic directly inside the Streamlit interface.

## Modularity

Each operation can be independently selected and configured.

## Interactivity

Important parameters are exposed through Streamlit controls.

## User Feedback

The interface communicates processing results and errors clearly.

## Maintainability

The project is structured so additional image processing operations can be added without redesigning the entire application.

---

# Deployment

The application is designed to be deployed as a public Streamlit application.

Deployment workflow:

```text
Local Development
       ↓
GitHub Repository
       ↓
Streamlit Deployment
       ↓
Public Web Application
```

The application can therefore be accessed without requiring users to install Python, OpenCV, or any other dependency locally.

---

# Deployment Checklist

Before deployment, verify:

* [ ] `app.py` is present
* [ ] `requirements.txt` is present
* [ ] `README.md` is present
* [ ] OpenCV dependencies are included
* [ ] Application runs locally
* [ ] Image upload works
* [ ] Processing operations work
* [ ] Shape detection works
* [ ] Pipeline mode works
* [ ] Processed images can be downloaded
* [ ] Application is publicly accessible

---

# Day 21 Internship Objectives

This project addresses the major objectives of the Day 21 task.

## Computer Vision Application Development

Combined multiple OpenCV techniques into a single application.

## User Interface Development

Built an interactive interface using Streamlit.

## Reusable Code

Organized image processing operations into reusable functions.

## Interactive Parameters

Added user-controlled parameters for different processing techniques.

## Deployment

Prepared the application for public Streamlit deployment.

## Documentation

Created professional project documentation explaining the architecture, workflow, and implementation.

---

# Learning Outcomes

Through this project, I developed practical experience with:

* OpenCV image processing
* NumPy image arrays
* Image channel management
* Grayscale conversion
* Gaussian filtering
* Canny edge detection
* Image transformations
* Image enhancement
* Contour detection
* Polygon approximation
* Geometric shape classification
* Streamlit application development
* Interactive UI controls
* Processing pipelines
* Python project organization
* Application deployment
* Git and GitHub workflow

---

# Future Improvements

The application can be extended with additional computer vision capabilities.

Potential improvements include:

* Object detection using YOLO
* Face detection
* Face recognition
* Image segmentation
* Perspective transformation
* Adaptive thresholding
* Histogram equalization
* Morphological operations
* Background removal
* Color-based object detection
* Batch image processing
* Video processing integration
* Webcam-based processing
* Advanced shape classification
* Custom computer vision models

---

# Project Context

This application was developed as part of the MLB Summer Internship Computer Vision track.

The project represents a progression from individual OpenCV exercises toward building a complete, reusable, and deployable computer vision application.

The earlier exercises focused on understanding individual techniques. Day 21 combines those concepts into a practical application with:

```text
Computer Vision
       +
Python
       +
OpenCV
       +
Interactive UI
       +
Streamlit
       +
Deployment
```

This progression demonstrates how fundamental computer vision algorithms can be transformed into a usable software product.

---

# Repository

**MLB Internship Repository**

`https://github.com/HadeedJalani/MLB-Internship`

Project location:

```text
MLB-Internship/
└── Day21/
```

---

# Conclusion

Computer Vision Image Processing Studio demonstrates how traditional computer vision techniques can be combined into a single interactive application.

The project goes beyond simply implementing individual OpenCV functions. It focuses on creating a practical workflow where users can:

* Upload an image
* Select an operation
* Configure processing parameters
* Preview the result
* Combine multiple operations
* Detect geometric shapes
* Download the processed image

The project also demonstrates the transition from experimental computer vision scripts to a structured and deployable application.

---

# Author

**Hadeed Jalani**

Computer Vision / Machine Learning Intern

Developed as part of the MLB Summer Internship Day 21.

<p align="center">

<strong>Computer Vision Image Processing Studio</strong>

<br>

Built with Python, OpenCV, NumPy, Pillow, and Streamlit.

</p>
