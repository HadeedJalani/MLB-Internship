# Computer Vision Image Processing Studio

A complete interactive Computer Vision application built with **Python, OpenCV, NumPy, PIL, and Streamlit**.

The application provides a unified environment for performing common image-processing and computer-vision operations through a clean web interface. Users can upload an image, configure processing parameters interactively, apply individual operations, or build a custom multi-step processing pipeline and download the final result.

---

## Project Overview

The **Computer Vision Image Processing Studio** was developed as part of **Day 21 of the MLB Summer Internship**.

The goal of this project is to combine the image-processing techniques learned throughout the previous days into a single practical application.

Instead of running individual Python scripts from the command line, this project provides an interactive Streamlit interface where users can experiment with different computer-vision operations and immediately observe their results.

The application supports both individual image transformations and chained processing pipelines, making it useful for learning, experimentation, and demonstrating fundamental Computer Vision concepts.

---

## Key Features

### Image Upload

Users can upload an image directly through the Streamlit interface.

Supported formats include:

- JPG
- JPEG
- PNG
- BMP
- WEBP

The uploaded image is displayed before processing so that the original result can be compared with the processed image.

---

### Single Operation Mode

The application allows users to select and apply individual image-processing operations.

Supported operations include:

1. Grayscale Conversion
2. Gaussian Blur
3. Canny Edge Detection
4. Image Rotation
5. Image Enhancement
6. Contour Detection
7. Shape Detection

Additional interactive controls are available for operations that require configurable parameters.

---

### Interactive Processing Controls

Processing parameters can be adjusted directly from the sidebar.

Examples include:

- Brightness
- Contrast
- Blur strength
- Rotation angle
- Canny lower threshold
- Canny upper threshold
- Enhancement parameters
- Shape detection sensitivity

Changes can be tested interactively without modifying the source code.

This makes the application suitable for experimentation and visual understanding of Computer Vision algorithms.

---

### Chain Multiple Filters

The application also supports a custom processing pipeline.

Users can select multiple operations and apply them sequentially to the same image.

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
Contour Detection
      ↓
Final Output