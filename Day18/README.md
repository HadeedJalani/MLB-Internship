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