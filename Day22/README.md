# 🔍 OCR Document Reader

<p align="center">
  <strong>Extract. Analyze. Understand.</strong><br>
  An AI-powered Optical Character Recognition application built with EasyOCR, OpenCV, and Streamlit.
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge\&logo=opencv\&logoColor=white)
![EasyOCR](https://img.shields.io/badge/EasyOCR-OCR-FF6F00?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)

</p>

---

## 📌 Overview

**OCR Document Reader** is a Computer Vision application that extracts machine-readable text from images using **Optical Character Recognition (OCR)**.

The application provides an interactive interface where users can upload an image, process it through an OCR pipeline, inspect the detected text and confidence scores, and export the extracted content as a text file.

The project focuses on understanding the complete OCR workflow rather than treating OCR as a single function call.

### Core Pipeline

```text
┌──────────────────┐
│   Input Image    │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Image Processing │
└────────┬─────────┘
         ↓
┌──────────────────┐
│  Text Detection  │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Text Recognition │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Confidence Score │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Extracted Text   │
└────────┬─────────┘
         ↓
┌──────────────────┐
│   Text Export    │
└──────────────────┘
```

---

## 🚀 Live Demo

> Replace the link below with your actual Streamlit deployment URL.

**🔗 [Launch OCR Document Reader](YOUR_STREAMLIT_APP_LINK)**

---

## 🎯 Project Objectives

This project was developed to gain practical experience with:

* Optical Character Recognition
* Computer Vision preprocessing
* Text detection and recognition
* OCR confidence analysis
* Image enhancement
* EasyOCR
* OpenCV
* Streamlit application development
* Model initialization and caching
* Text extraction and export
* Real-world OCR limitations

---

# 🧠 What is OCR?

**Optical Character Recognition (OCR)** is a Computer Vision technique used to convert text contained inside images into machine-readable data.

For example, an image may contain:

```text
Invoice Number: 10245
Customer: Hadeed Jalani
Total Amount: $150
Date: 2026-08-10
```

OCR converts the visual information into text that a computer can store, search, analyze, or process:

```text
Invoice Number: 10245
Customer: Hadeed Jalani
Total Amount: $150
Date: 2026-08-10
```

This makes OCR useful for applications such as:

* Document digitization
* Receipt processing
* Invoice extraction
* Identity document processing
* Sign recognition
* Searchable archives
* Automated data entry
* Document management systems

---

# ⚙️ How the OCR Pipeline Works

## 1. Image Input

The user uploads an image containing text.

Supported formats depend on the image-processing pipeline and commonly include:

```text
JPG
JPEG
PNG
BMP
WEBP
```

---

## 2. Image Preprocessing

Image quality has a major impact on OCR performance.

Depending on the input, preprocessing can be used to improve the visibility and separation of text.

Common techniques include:

* Grayscale conversion
* Contrast enhancement
* Thresholding
* Noise reduction
* Image resizing
* Upscaling

The purpose is to make the visual characteristics of the text easier for the OCR model to interpret.

---

## 3. Text Detection

EasyOCR identifies regions of the image that are likely to contain text.

The OCR engine returns information such as:

* Text region
* Recognized text
* Confidence score

Conceptually:

```text
Image
  │
  ├── Text Region 1 → "Invoice" → 0.97
  ├── Text Region 2 → "Number"  → 0.94
  └── Text Region 3 → "10245"   → 0.91
```

---

## 4. Text Recognition

Once text regions have been detected, EasyOCR recognizes the characters contained within those regions.

The result is converted into machine-readable text.

---

## 5. Confidence Analysis

OCR systems provide a confidence value representing how strongly the model believes in a particular recognition result.

For example:

| Detected Text | Confidence |
| ------------- | ---------: |
| Invoice       |        97% |
| Number        |        94% |
| 10245         |        91% |

An overall confidence metric can also be calculated to provide a quick indication of OCR reliability.

> Higher confidence generally indicates greater certainty, but it should not be treated as a guarantee of correctness.

---

## 6. Text Export

After reviewing the OCR output, the extracted text can be exported as a `.txt` file.

Example:

```text
document_ocr.txt
```

This makes the extracted content easy to store, edit, or process further.

---

# 🧰 OCR Libraries Explored

Several OCR libraries and frameworks were investigated during the development process.

| Library           | Strengths                                            | Typical Applications    |
| ----------------- | ---------------------------------------------------- | ----------------------- |
| **Tesseract OCR** | Mature, lightweight, widely used                     | Clean printed documents |
| **EasyOCR**       | Simple API, deep-learning based, multilingual        | General-purpose OCR     |
| **PaddleOCR**     | Strong accuracy and document-processing capabilities | Advanced document OCR   |
| **DocTR**         | Deep-learning document OCR pipeline                  | Document analysis       |

---

# ⭐ Why EasyOCR?

EasyOCR was selected for the implementation because it provides a practical balance between **simplicity, capability, and integration speed**.

### Advantages

* Easy Python integration
* Deep-learning-based OCR
* Supports multiple languages
* Performs text detection and recognition
* Straightforward API
* Suitable for interactive applications
* Can utilize GPU acceleration when available

For an internship-level Computer Vision project, EasyOCR provides a good foundation for understanding modern OCR workflows without introducing unnecessary system complexity.

---

# 🖼️ Image Preprocessing

One of the most important concepts explored in this project is that:

> **OCR accuracy is strongly influenced by image quality.**

The same OCR model can produce significantly different results depending on the input image.

### Grayscale

Converts a color image into a single-channel intensity representation.

```text
RGB Image
   ↓
Grayscale Image
```

This can simplify subsequent image-processing operations.

### Contrast Enhancement

Improves the visual separation between foreground text and its background.

### Thresholding

Converts an image into a binary representation.

This can be particularly effective for certain high-contrast document images.

### Otsu Thresholding

Automatically estimates a threshold based on the distribution of pixel intensities.

### Upscaling

Increasing the resolution of small text can sometimes improve recognition.

However, excessive upscaling can increase computation without necessarily improving accuracy.

---

# 📊 OCR Challenges

Real-world OCR is considerably more difficult than recognizing clean, high-resolution printed text.

## ✍️ Handwriting

Handwritten text can be difficult because:

* Characters have inconsistent shapes.
* Letters can connect.
* Spacing varies.
* Writing styles differ significantly.

## 📉 Low Image Quality

Blur, compression artifacts, and low resolution can make characters ambiguous.

For example:

```text
O → 0
I → 1
S → 5
```

may occasionally be confused.

## 💡 Uneven Lighting

Shadows, reflections, and inconsistent illumination can make parts of a document difficult to recognize.

## 🖼️ Complex Backgrounds

Background patterns or objects may be incorrectly interpreted as text.

## 🔤 Unusual Fonts

Decorative or stylized fonts can reduce OCR accuracy.

## 🔎 Small Text

Very small characters may not contain enough visual information for reliable recognition.

## 📐 Perspective Distortion

Photographs taken from an angle can distort text and make recognition more difficult.

---

# ⚡ Performance Considerations

OCR models can be computationally expensive.

Processing time may depend on:

* Image resolution
* Number of text regions
* CPU/GPU availability
* OCR model
* Preprocessing operations
* Image complexity

The OCR reader/model can therefore be cached within Streamlit so that it does not need to be unnecessarily initialized during every interaction.

This is particularly important for Streamlit applications because the application script can rerun when users interact with widgets.

---

# 🧵 Multi-Threading & Hardware Acceleration

OCR frameworks use different approaches to parallel processing and hardware acceleration.

### Tesseract

Tesseract generally processes images individually. Parallel processing can be implemented at the application level when processing multiple independent images.

### EasyOCR

EasyOCR is built on PyTorch and can utilize the underlying framework's CPU/GPU capabilities.

GPU acceleration can significantly improve inference performance when compatible hardware is available.

### PaddleOCR

PaddleOCR provides strong support for optimized inference and hardware acceleration, making it attractive for larger-scale OCR systems.

### DocTR

DocTR is built around deep-learning frameworks and can also benefit from hardware acceleration.

### Practical Consideration

Parallel OCR processing should be implemented carefully.

Running too many OCR workers simultaneously can increase:

* Memory usage
* CPU contention
* GPU memory consumption

Therefore, more threads do not automatically mean better performance.

For this project, the primary focus remained on **reliable OCR functionality and application usability** rather than large-scale parallel processing.

---

# ✨ Application Features

### 📤 Image Upload

Upload an image containing text directly through the Streamlit interface.

### 🔍 OCR Extraction

Run EasyOCR to detect and recognize text from the uploaded image.

### 🧠 Image Processing

Apply image-processing techniques where appropriate to improve OCR input quality.

### 📦 Text Detection

Identify regions of the image containing text.

### 📈 Confidence Analysis

Inspect OCR confidence scores for detected text.

### 📝 Extracted Text

View the recognized text directly within the application.

### 💾 Text Export

Download the extracted result as a `.txt` file.

---

# 🏗️ Project Structure

```text
Day22/
│
├── app.py
├── ocr_utils.py
├── requirements.txt
├── README.md
│
├── input_images/
│   ├── document.jpg
│   ├── receipt.jpg
│   └── ...
│
└── extracted_text/
    ├── document_ocr.txt
    └── ...
```

> Update the structure above if your actual Day22 files differ.

---

# 🛠️ Technology Stack

| Technology    | Purpose                              |
| ------------- | ------------------------------------ |
| **Python**    | Core programming language            |
| **EasyOCR**   | Optical Character Recognition        |
| **OpenCV**    | Image processing and preprocessing   |
| **NumPy**     | Numerical and image-array operations |
| **Pillow**    | Image handling                       |
| **Streamlit** | Interactive web application          |

---

# 💻 Installation

## 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

## 2. Navigate to Day 22

```bash
cd MLB-Internship/Day22
```

## 3. Create a Virtual Environment

```bash
python -m venv venv
```

## 4. Activate the Environment

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

## 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

Open the address in your browser and upload an image containing text.

---

# 🔄 Usage Workflow

```text
1. Upload Image
        ↓
2. Configure Processing
        ↓
3. Preview Image
        ↓
4. Run OCR
        ↓
5. Review Detected Text
        ↓
6. Analyze Confidence
        ↓
7. Export Result
```

---

# 🧪 Testing

The OCR pipeline can be tested using different types of images, including:

* Printed documents
* Receipts
* Signs
* Book pages
* Handwritten notes
* Different lighting conditions
* Different text sizes
* Images with complex backgrounds

Testing across different image categories demonstrates an important OCR principle:

> **There is no universal preprocessing technique that produces optimal results for every image.**

The best approach depends on:

* Image quality
* Lighting
* Contrast
* Resolution
* Font
* Background
* Text orientation
* Text size

---

# 📚 Key Learning Outcomes

Through this project, I gained practical experience with:

* Optical Character Recognition
* EasyOCR
* Computer Vision preprocessing
* OpenCV
* Text detection
* Text recognition
* Bounding boxes
* Confidence scores
* Image enhancement
* Streamlit development
* Model caching
* OCR performance considerations
* Real-world OCR limitations

---

# 🔮 Future Improvements

The project can be extended into a more advanced document-intelligence system.

Potential improvements include:

* 🌍 Multi-language OCR
* 📄 PDF document support
* 📚 Batch image processing
* ✏️ Editable OCR results
* 📊 CSV/JSON export
* 🔍 Searchable document generation
* 📐 Automatic perspective correction
* 🤖 Automatic preprocessing selection
* ⚡ GPU acceleration
* 🧠 PaddleOCR integration
* 🧠 DocTR integration
* 📑 Document layout analysis
* 📊 Table detection
* ✍️ Specialized handwriting recognition
* 🗂️ Structured document information extraction

---

# 📈 From OCR to Document Intelligence

This project provides a foundation for more advanced AI systems.

A future version could transform:

```text
Image
  ↓
OCR
  ↓
Text
  ↓
Information Extraction
  ↓
Structured Data
  ↓
Database / API
```

For example, an invoice could eventually be converted from an image into:

```json
{
  "invoice_number": "10245",
  "date": "2026-08-10",
  "total": 150,
  "currency": "USD"
}
```

This demonstrates how OCR can become the first stage of a broader **Document AI pipeline**.

---

# 🎓 Internship Context

This project was completed as part of the **MLB Summer Internship — Computer Vision learning track**.

It represents a progression from fundamental image-processing concepts toward practical AI-powered applications capable of working with real-world visual data.

---

# 👨‍💻 Author

## Hadeed Jalani

**Final-Year BSCS Student | AI/ML & Full-Stack Developer**

Focused on building practical applications using:

```text
Python • AI/ML • Computer Vision • Full-Stack Development
```

---

<p align="center">

### 🚀 Day 22  Optical Character Recognition

<strong>Built with Python · EasyOCR · OpenCV · Streamlit</strong>

</p>
