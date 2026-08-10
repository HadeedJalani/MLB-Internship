# Day 17 — Image Transformations & Enhancement 🖼️

![Python](https://img.shields.io/badge/Python-3.x-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![Streamlit](https://img.shields.io/badge/Deployment-Streamlit-red)

Part of the **MLBench Summer Internship Program**.

This module builds on the OpenCV fundamentals covered in Day 16 and focuses on **image transformations, image enhancement, and document preprocessing**.

The main goal of Day 17 was to understand how images can be geometrically transformed and visually enhanced before being used in Computer Vision applications such as OCR, document analysis, object detection, and image classification.

---

# 📌 Overview

| Task | Image Transformations & Enhancement |
|---|---|
| Framework | OpenCV |
| Language | Python |
| Input | Document Images |
| Transformations | Translation, Rotation, Scaling, Affine, Perspective |
| Enhancement | Brightness, Contrast, Blur, Denoising, Sharpening |
| Mini Project | Document Image Enhancement Tool |
| Deployment | Streamlit |
| Dataset | 10 Document Images |

---

# 🧠 Concepts Covered

## Image Transformations

The project implements the following geometric transformations:

- Translation
- Rotation
- Scaling
- Affine Transformation
- Perspective Transformation

These operations modify the position, orientation, size, or perspective of an image.

---

# 🔄 Translation

Translation moves an image horizontally and/or vertically without changing its shape.

OpenCV implementation:

```python
cv2.warpAffine()