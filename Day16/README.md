# Day 16 – OpenCV Fundamentals & Basic Image Processing

## Project Overview

This project was completed as part of the **MLB Summer Internship – Day 16**. The objective was to understand the fundamentals of OpenCV and build a reusable Image Processing Toolkit capable of performing common image manipulation tasks through both Python scripts and a Streamlit web application.

The application allows users to upload an image, apply different image processing operations, visualize the results instantly, and download the processed image.

---

# Objectives

- Understand how images are represented in OpenCV.
- Learn basic image processing techniques.
- Build reusable image processing functions.
- Create an interactive Streamlit application.
- Organize outputs in a structured project.

---

# Project Structure

```
Day16/
│
├── input_images/
│   ├── landscape.jpg
│   ├── person.jpg
│   ├── vehicle.jpg
│   ├── document.jpg
│   └── object.jpg
│
├── output_images/
│   ├── grayscale/
│   ├── resized/
│   ├── cropped/
│   ├── rotated/
│   ├── flipped/
│   ├── drawings/
│   └── toolkit_outputs/
│
├── opencv_fundamentals.py
├── basic_image_operations.py
├── drawing_shapes.py
├── image_processing_toolkit.py
├── streamlit_app.py
├── requirements.txt
└── README.md
```

---

# Features Implemented

## OpenCV Fundamentals

- Read image
- Display image
- Save image
- Extract image dimensions
- Display image properties
- Convert BGR to RGB
- Convert image to grayscale

---

## Basic Image Processing

- Resize image
- Crop image
- Rotate image
- Flip image horizontally
- Flip image vertically

---

## Drawing Operations

- Draw Rectangle
- Draw Circle
- Draw Line
- Draw Polygon
- Add Custom Text

---

## Image Processing Toolkit

The toolkit supports:

- Upload image
- Resize
- Crop
- Rotate
- Flip
- Draw Shapes
- Add Text
- Convert to Grayscale
- Brightness Adjustment
- Contrast Adjustment
- RGB vs BGR Comparison
- Download Processed Image
- Undo Previous Operation
- Reset Image

---

# Difference Between BGR and RGB

OpenCV stores images in **BGR (Blue, Green, Red)** format by default, whereas most visualization libraries such as Matplotlib and Pillow use **RGB (Red, Green, Blue)**.

Although both formats contain the same color information, the order of the color channels is different. If a BGR image is displayed directly using libraries expecting RGB, the colors appear incorrect. Therefore, images are often converted from BGR to RGB before displaying.

---

# What are Grayscale Images?

A grayscale image contains only intensity values instead of three color channels.

Each pixel represents a shade ranging from black to white.

Grayscale images are commonly used because:

- They reduce computational complexity.
- They require less memory.
- Many computer vision algorithms perform better on grayscale images.
- Edge detection and feature extraction become simpler.

---

# OpenCV Functions Used

Some of the major OpenCV functions used in this project include:

- `cv2.imread()`
- `cv2.imshow()`
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

# Challenges Faced

During development, several challenges were encountered:

- Understanding the difference between BGR and RGB color spaces.
- Managing image updates while performing multiple operations.
- Preserving image quality after repeated processing.
- Implementing an Undo feature using Streamlit Session State.
- Making the application deployment-compatible by replacing `opencv-python` with `opencv-python-headless`.

These challenges were resolved by carefully managing session state, organizing reusable helper functions, and following Streamlit deployment best practices.

---

# Technologies Used

- Python
- OpenCV
- Streamlit
- NumPy
- Pillow

---

# Learning Outcomes

By completing this project, I learned:

- How digital images are represented in OpenCV.
- Practical image manipulation techniques.
- Color space conversions.
- Drawing and annotation using OpenCV.
- Building interactive image processing applications with Streamlit.
- Managing application state and deployment.

---

# Future Improvements

Possible enhancements include:

- Interactive drawing using mouse events.
- Image filters (Blur, Sharpen, Edge Detection).
- Histogram visualization.
- Image comparison mode.
- Batch image processing.
- Undo/Redo history with multiple levels.
- Support for additional image formats.

---

# Author

**Hadeed Jalani**

MLB Summer Internship – Day 16

OpenCV Fundamentals & Image Processing Toolkit