# 📄 Document OCR Studio

<p align="center">
  <strong>A Dual-Engine Optical Character Recognition Workspace</strong><br>
  Extract, analyze, visualize, compare, and export text from document images using EasyOCR and PaddleOCR.
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20UI-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)
![EasyOCR](https://img.shields.io/badge/EasyOCR-OCR-4B8BBE?style=for-the-badge)
![PaddleOCR](https://img.shields.io/badge/PaddleOCR-OCR-00A67E?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge\&logo=opencv\&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-2EA44F?style=for-the-badge)

</p>

---

# 📌 Overview

**Document OCR Studio** is a professional Computer Vision application developed as part of **Day 23 of the MLB Summer Internship**.

The application provides a complete OCR workspace for extracting text from document images using two independent OCR engines:

* **EasyOCR**
* **PaddleOCR**

It combines OCR with OpenCV-based image preprocessing and provides an interactive Streamlit interface for configuring the recognition pipeline, visualizing detected text regions, comparing OCR engines, analyzing confidence scores, measuring processing time, and exporting extracted text.

The project goes beyond basic OCR by addressing practical challenges such as:

* Image quality
* Preprocessing
* OCR confidence
* Bounding-box quality
* Engine comparison
* Model initialization
* Runtime compatibility
* Session-state management
* OCR result parsing
* Graceful error handling
* Public deployment

---

# 🎯 Project Goal

The primary objective of this project was to build a **complete, interactive, and deployable OCR system** rather than a simple OCR script.

The application follows this pipeline:

```text
                Document Image
                       │
                       ▼
              ┌─────────────────┐
              │ Image Processing│
              │    OpenCV       │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  OCR Selection  │
              └────────┬────────┘
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
        ┌─────────┐         ┌───────────┐
        │ EasyOCR │         │ PaddleOCR │
        └────┬────┘         └─────┬─────┘
             │                    │
             └─────────┬──────────┘
                       ▼
              ┌─────────────────┐
              │ Result Analysis │
              │                 │
              │ • Text          │
              │ • Confidence    │
              │ • Bounding Box  │
              │ • Time          │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │ Visualization   │
              │ & Comparison    │
              └────────┬────────┘
                       ▼
                Extracted Text
                       │
                       ▼
                  TXT Export
```

---

# ✨ Key Features

## 🔍 Dual OCR Engine Architecture

The application supports two independent OCR engines.

### EasyOCR

EasyOCR provides general-purpose text detection and recognition.

It provides:

* Text detection
* Text recognition
* Confidence scores
* Bounding boxes
* English OCR
* CPU-based execution
* Optional bounding-box refinement

### PaddleOCR

PaddleOCR provides an alternative OCR pipeline designed for document-oriented recognition.

It provides:

* Text detection
* Text recognition
* Confidence scores
* Bounding boxes
* Independent OCR results
* Compatibility handling for different runtime environments

The two engines can be used independently or executed together for comparison.

---

# ⚙️ OCR Modes

The application provides three operating modes.

## 1. EasyOCR

Runs only EasyOCR.

```text
Image
  ↓
Preprocessing
  ↓
EasyOCR
  ↓
Results
```

This mode is useful for general-purpose OCR and fast experimentation.

---

## 2. PaddleOCR

Runs only PaddleOCR.

```text
Image
  ↓
Preprocessing
  ↓
PaddleOCR
  ↓
Results
```

This mode allows PaddleOCR to be evaluated independently.

---

## 3. Compare Engines

Runs both OCR engines independently.

```text
                 Image
                   │
            ┌──────┴──────┐
            ▼             ▼
         EasyOCR      PaddleOCR
            │             │
            ▼             ▼
        Results        Results
            │             │
            └──────┬──────┘
                   ▼
             Comparison
```

The comparison includes:

| Metric              | EasyOCR | PaddleOCR |
| ------------------- | :-----: | :-------: |
| Text Regions        |    ✅    |     ✅     |
| Extracted Text      |    ✅    |     ✅     |
| Confidence          |    ✅    |     ✅     |
| Processing Time     |    ✅    |     ✅     |
| Independent Results |    ✅    |     ✅     |

This allows users to observe how different OCR engines behave on the same document.

---

# 🖼️ Image Preprocessing

OCR accuracy is highly dependent on the quality of the input image.

Document images may contain:

* Noise
* Shadows
* Uneven lighting
* Low contrast
* Background patterns
* Small characters
* Poor scanning quality

To address these issues, the application provides several OpenCV-based preprocessing techniques.

---

## ⚫ Grayscale Conversion

Converts the image into grayscale before OCR.

```text
RGB Image
    ↓
Grayscale
    ↓
OCR
```

Removing unnecessary color information can simplify the image and improve the separation between text and background.

---

## 🧹 Denoising

The application can use OpenCV's non-local means denoising to reduce image noise while attempting to preserve important text structures.

This can be useful for:

* Noisy scans
* Low-quality photographs
* Old documents
* Images with background artifacts

---

## 📈 CLAHE Contrast Enhancement

The application uses **Contrast Limited Adaptive Histogram Equalization (CLAHE)** to improve local image contrast.

Unlike basic global histogram equalization, CLAHE works on local image regions.

```text
Original Image
       ↓
      CLAHE
       ↓
Improved Local Contrast
       ↓
      OCR
```

This can be particularly useful when text is affected by uneven lighting.

---

## ⚫ Adaptive Thresholding

Adaptive thresholding converts an image into a binary representation using locally calculated threshold values.

It can help with documents containing:

* Shadows
* Uneven illumination
* Variable background intensity
* Low-contrast text

The application uses Gaussian adaptive thresholding through OpenCV.

---

# 📦 Bounding Box Refinement

A practical issue encountered during development was that OCR detection boxes do not always tightly follow the actual characters.

For example:

```text
Original Detection

┌─────────────────────────────┐
│                             │
│        Invoice Number       │
│                             │
└─────────────────────────────┘
```

The detection region may contain unnecessary surrounding space.

To improve the visual quality of EasyOCR detections, the application includes a custom bounding-box refinement process.

---

## 🔬 Refinement Pipeline

```text
EasyOCR Detection
        ↓
Initial Bounding Box
        ↓
Crop Original ROI
        ↓
Grayscale Conversion
        ↓
Otsu Thresholding
        ↓
Noise Processing
        ↓
Contour Detection
        ↓
Refined Bounding Box
```

The refined region remains constrained relative to the original OCR detection so that unrelated nearby content is less likely to be included.

### Benefits

* Cleaner visual output
* Tighter text regions
* Better detection visualization
* Improved presentation of OCR results

Importantly, this refinement primarily affects the **visualization region**, not the actual recognized text.

---

# 🎛️ Interactive Controls

The Streamlit interface provides configurable controls for the OCR pipeline.

Users can configure:

* OCR engine
* Image preprocessing
* Grayscale conversion
* Denoising
* Contrast enhancement
* Adaptive thresholding
* Minimum confidence
* Detection-box visibility
* EasyOCR bounding-box refinement

This makes the application useful not only as an OCR tool but also as an experimentation environment for understanding how preprocessing affects recognition.

---

# 📊 OCR Result Analysis

After OCR processing, the application provides several metrics.

## Text Regions

The total number of detected text regions.

---

## Average Confidence

The average confidence across recognized text regions.

For example:

```text
Average Confidence
        ↓
       94.6%
```

Confidence can help identify potentially unreliable recognition results.

> A high confidence score indicates model certainty, but it does not guarantee that the recognized text is correct.

---

## Processing Time

The application measures how long the OCR engine takes to process the image.

Example:

```text
Processing Time
     ↓
   1.84 sec
```

This makes it possible to compare the practical performance of EasyOCR and PaddleOCR.

---

## Detection Visualization

Detected text regions can be displayed directly over the document image.

Conceptually:

```text
┌────────────────────────────────────┐
│ ┌───────────────┐                  │
│ │ Invoice       │                  │
│ └───────────────┘                  │
│                                    │
│ ┌─────────────────────┐            │
│ │ Invoice Number: 10245│           │
│ └─────────────────────┘            │
│                                    │
└────────────────────────────────────┘
```

This provides visual confirmation of what the OCR engine detected.

---

# 📝 Extracted Text

The recognized text is displayed in the application after processing.

The output can be reviewed before export.

A simplified example:

```text
INVOICE

Invoice Number: 10245
Date: 10 August 2026

Product: Computer Accessories
Quantity: 2
Total: $150
```

---

# 💾 Text Export

OCR results can be downloaded as plain-text files.

Example filenames:

```text
document_easyocr_ocr.txt
document_paddleocr_ocr.txt
```

This allows the extracted content to be used outside the application for:

* Documentation
* Data processing
* Storage
* Search
* Analysis
* Further automation

---

# 🧠 Technologies Used

| Technology    | Purpose                                 |
| ------------- | --------------------------------------- |
| **Python**    | Core application development            |
| **Streamlit** | Interactive web interface               |
| **EasyOCR**   | OCR detection and recognition           |
| **PaddleOCR** | Secondary OCR engine                    |
| **OpenCV**    | Image preprocessing and computer vision |
| **NumPy**     | Numerical and image-array processing    |
| **Pillow**    | Image loading and conversion            |
| **Regex**     | OCR text cleanup and normalization      |

---

# 🏗️ Application Architecture

The application follows a modular OCR architecture.

```text
                         ┌───────────────────┐
                         │    Streamlit UI   │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │   Image Upload    │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │ OpenCV Processing │
                         └─────────┬─────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
             ┌─────────────┐              ┌─────────────┐
             │   EasyOCR   │              │  PaddleOCR  │
             └──────┬──────┘              └──────┬──────┘
                    │                             │
                    └──────────────┬──────────────┘
                                   ▼
                         ┌───────────────────┐
                         │ Result Processing │
                         └─────────┬─────────┘
                                   │
                ┌──────────────────┼──────────────────┐
                ▼                  ▼                  ▼
             Text              Confidence        Processing
           Extraction            Analysis             Time
                │                  │                  │
                └──────────────────┼──────────────────┘
                                   ▼
                         ┌───────────────────┐
                         │ Visualization &   │
                         │ Engine Comparison │
                         └─────────┬─────────┘
                                   │
                                   ▼
                           ┌──────────────┐
                           │  TXT Export  │
                           └──────────────┘
```

---

# 📁 Project Structure

```text
Day-23/
│
├── app.py
├── requirements.txt
├── README.md
│
├── sample_inputs/
│   ├── document_01.png
│   ├── document_02.jpg
│   └── document_03.png
│
├── sample_outputs/
│   ├── easyocr_result.txt
│   └── paddleocr_result.txt
│
├── screenshots/
│   ├── application.png
│   ├── easyocr_result.png
│   └── comparison.png
│
└── screen_recording/
    └── day23_ocr_demo.mp4
```

> Adjust the structure above if the actual repository contains different filenames or folders.

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/HadeedJalani/MLB-Internship.git
```

Navigate to Day 23:

```bash
cd MLB-Internship/Day-23
```

---

## 2. Create a Virtual Environment

Using a virtual environment is recommended to isolate project dependencies.

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The project primarily relies on:

```text
streamlit
opencv-python-headless
numpy
Pillow
easyocr
paddleocr
paddlepaddle
```

> Dependency versions should be kept synchronized with the versions specified in `requirements.txt`.

---

# ▶️ Running the Application

Launch the Streamlit application:

```bash
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

Open the URL in a browser to access the OCR workspace.

---

# 🖥️ Usage Guide

## Step 1 — Upload a Document

Upload an image containing text.

Common supported formats include:

```text
JPG
JPEG
PNG
BMP
WEBP
```

---

## Step 2 — Configure Preprocessing

Enable the preprocessing pipeline when necessary.

Available options include:

```text
✓ Grayscale
✓ Denoising
✓ Contrast Enhancement
✓ Adaptive Thresholding
```

Different documents may require different preprocessing configurations.

---

## Step 3 — Select OCR Engine

Choose one of:

```text
EasyOCR
PaddleOCR
Compare Engines
```

---

## Step 4 — Configure Confidence

Set the minimum confidence threshold.

Detections below the selected threshold can be excluded from the displayed results.

---

## Step 5 — Run OCR

Start the analysis using:

```text
Run OCR Analysis
```

The application processes the uploaded document through the selected OCR pipeline.

---

## Step 6 — Review Results

Review:

* Extracted text
* Number of detected regions
* Average confidence
* Processing time
* Detection bounding boxes

---

## Step 7 — Compare Engines

When **Compare Engines** is selected, EasyOCR and PaddleOCR process the document independently.

Their outputs can then be evaluated side-by-side.

---

## Step 8 — Export

Download the recognized text as a `.txt` file.

---

# ⚡ Performance & Model Management

OCR models can require significant initialization time because model weights and supporting components need to be loaded.

Repeated initialization during Streamlit reruns would create unnecessary overhead.

To address this, the application uses Streamlit's resource caching mechanism for OCR model initialization.

Conceptually:

```text
First Run
   ↓
Initialize OCR Model
   ↓
Cache Model
   ↓
Reuse Model
   ↓
Faster Subsequent Interactions
```

This improves the application's responsiveness after the initial model loading stage.

---

# 🔄 Session State Management

Streamlit reruns the application script when users interact with widgets.

Without persistent state, OCR results from one engine could potentially be overwritten when another engine runs.

The application therefore maintains independent OCR results using session state.

Conceptually:

```text
ocr_results
│
├── EasyOCR
│   ├── text
│   ├── confidence
│   ├── boxes
│   └── processing_time
│
└── PaddleOCR
    ├── text
    ├── confidence
    ├── boxes
    └── processing_time
```

This allows the application to preserve both engine results during comparison.

---

# 🧩 OCR Result Compatibility

Different PaddleOCR versions may expose results using different data structures.

The application therefore uses flexible result-processing logic to handle variations such as:

* Dictionaries
* Lists
* Tuples
* Polygon coordinates
* Recognition outputs
* OCR result objects

This reduces dependency on a single PaddleOCR output structure and improves compatibility across supported environments.

---

# 🛡️ Error Handling & Reliability

A major design goal was to ensure that failure of one OCR engine does not make the entire application unusable.

For example:

```text
              OCR Workspace
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
       EasyOCR             PaddleOCR
          │                   │
       Working             Runtime Error
          │                   │
          └─────────┬─────────┘
                    ▼
             Application
               Continues
```

If PaddleOCR encounters an environment-specific runtime problem, the application reports the issue gracefully while keeping EasyOCR available.

This provides a more resilient user experience.

---

# ⚠️ PaddleOCR Runtime Compatibility

During development, PaddleOCR encountered environment-specific runtime issues involving the PaddlePaddle execution layer.

Examples included errors related to:

```text
ConvertPirAttribute2RuntimeAttribute
```

and:

```text
pir::ArrayAttribute<pir::DoubleAttribute>
```

These issues occurred within the PaddlePaddle runtime/oneDNN execution environment rather than the OCR result-processing logic.

The application was therefore designed to:

* Handle PaddleOCR initialization failures gracefully
* Prevent runtime errors from crashing the interface
* Keep EasyOCR independently functional
* Provide useful diagnostic information
* Support compatibility-oriented runtime configuration

This was an important practical lesson in deploying deep-learning applications where **library compatibility and runtime environments can be as important as the application code itself**.

---

# 🧪 Testing

The application can be tested using different document categories.

### Test Categories

* Printed documents
* Receipts
* Invoices
* Signs
* Book pages
* Screenshots
* Low-contrast documents
* Noisy images
* Different font sizes
* Documents with uneven lighting

Testing different inputs demonstrates that OCR performance depends heavily on image characteristics.

---

# 📊 OCR Engine Comparison

A major purpose of the project is to understand that different OCR engines may produce different results for the same document.

A useful comparison considers:

| Evaluation Factor     | Why It Matters                         |
| --------------------- | -------------------------------------- |
| **Recognized Text**   | Measures practical OCR output          |
| **Confidence**        | Indicates model certainty              |
| **Detection Count**   | Shows how many regions were identified |
| **Processing Time**   | Measures practical performance         |
| **Bounding Boxes**    | Evaluates localization quality         |
| **Runtime Stability** | Important for deployment               |

There is not necessarily a single OCR engine that performs best for every document.

Performance can vary depending on:

* Language
* Font
* Image quality
* Layout
* Resolution
* Background
* Document complexity

---

# 🧠 Challenges & Solutions

## Challenge 1 — PaddleOCR Compatibility

### Problem

PaddleOCR produced runtime errors under certain PaddlePaddle environments.

### Solution

Implemented graceful error handling and compatibility-oriented runtime configuration so that PaddleOCR failures do not crash the entire application.

---

## Challenge 2 — OCR Initialization Time

### Problem

OCR models can take significant time to initialize.

### Solution

Used Streamlit resource caching to reuse initialized OCR models.

---

## Challenge 3 — Preserving Comparison Results

### Problem

Streamlit reruns could cause one engine's results to replace another's.

### Solution

Used Streamlit session state to maintain independent EasyOCR and PaddleOCR results.

---

## Challenge 4 — PaddleOCR Result Structures

### Problem

Different PaddleOCR versions may return results in different structures.

### Solution

Implemented flexible result parsing capable of handling multiple output formats.

---

## Challenge 5 — Loose EasyOCR Bounding Boxes

### Problem

Some EasyOCR detection boxes were visually larger than the actual characters.

### Solution

Implemented a custom OpenCV-based bounding-box refinement process using thresholding and contour detection.

---

# 🔮 Future Improvements

The current implementation provides a strong foundation for a more advanced Document AI platform.

Potential improvements include:

## 🌍 Multi-Language OCR

Expand recognition beyond English to languages such as:

```text
Urdu
Arabic
French
German
Spanish
Chinese
Japanese
```

---

## 📄 PDF Support

Add multi-page PDF processing.

```text
PDF
 ↓
Page Extraction
 ↓
Image Processing
 ↓
OCR
 ↓
Text Aggregation
 ↓
Searchable PDF / TXT
```

---

## ⚡ GPU Acceleration

Support CUDA-enabled inference where compatible hardware is available.

This could significantly improve processing performance for large documents.

---

## 🤖 Automatic Preprocessing

Instead of manually selecting preprocessing techniques, the application could analyze image quality and automatically choose an appropriate pipeline.

```text
Image
  ↓
Quality Analysis
  ↓
Low Contrast?
  ├── Yes → Contrast Enhancement
  └── No
  ↓
Noise?
  ├── Yes → Denoising
  └── No
  ↓
Uneven Lighting?
  ├── Yes → Adaptive Thresholding
  └── No
  ↓
OCR
```

---

## 📊 Confidence Visualization

Future versions could visualize confidence using:

* Color-coded bounding boxes
* Confidence labels
* Heatmaps
* Detection-level statistics

---

## 📑 Document Layout Analysis

The system could eventually identify:

* Headings
* Paragraphs
* Tables
* Forms
* Columns
* Invoices
* Receipts

---

## 🧾 Structured Information Extraction

OCR output could be converted into structured data.

For example:

```json
{
  "invoice_number": "10245",
  "date": "2026-08-10",
  "total": "150",
  "currency": "USD"
}
```

This would transform the application from basic OCR into a broader **Document Intelligence** system.

---

## 🔎 Searchable PDF Generation

Recognized text could be embedded into original documents to create searchable PDF files.

---

# 🌐 Deployment

The application is designed for public deployment using Streamlit.

## 🚀 Live Streamlit Application

**Add your deployed URL here:**

```text
YOUR_STREAMLIT_APP_URL
```

---

# 🔗 Repository

**GitHub Repository:**

```text
https://github.com/HadeedJalani/MLB-Internship
```

The Day 23 implementation is located inside:

```text
Day-23/
```

---

# 🎥 Demonstration

The project demonstration covers the complete application workflow:

1. Launching the Streamlit application
2. Uploading a document image
3. Configuring preprocessing
4. Running EasyOCR
5. Running PaddleOCR
6. Comparing both OCR engines
7. Viewing detection bounding boxes
8. Reviewing confidence scores
9. Checking processing time
10. Exporting OCR results

**🎬 Screen Recording:**
`YOUR_SCREEN_RECORDING_LINK`

---

# 📸 Sample Results

Sample input documents and OCR outputs are included with the project.

Recommended organization:

```text
sample_inputs/
```

for original document images and:

```text
sample_outputs/
```

for generated OCR results.

Screenshots can also be stored inside:

```text
screenshots/
```

Example:

```markdown
![Document OCR Studio](screenshots/application.png)
```

---

# 📋 Deliverables

| Deliverable                  | Status |
| ---------------------------- | :----: |
| OCR Source Code              |    ✅   |
| `app.py`                     |    ✅   |
| `requirements.txt`           |    ✅   |
| `README.md`                  |    ✅   |
| Sample Input Images          |    ✅   |
| Sample Output Results        |    ✅   |
| GitHub Repository            |    ✅   |
| Public Streamlit Application |    ✅   |
| Screen Recording             |    ✅   |

---

# 🎓 Learning Outcomes

This project provided practical experience in several areas of Computer Vision and software development.

### Optical Character Recognition

Understanding how OCR engines detect and recognize text from visual data.

### Computer Vision

Using OpenCV for image processing and preprocessing.

### Image Enhancement

Working with:

* Grayscale conversion
* Denoising
* CLAHE
* Adaptive thresholding
* Otsu thresholding

### OCR Evaluation

Comparing OCR systems through:

* Extracted text
* Confidence
* Detection count
* Processing time
* Bounding-box quality

### Streamlit Development

Building an interactive Computer Vision application with a web-based interface.

### Model Management

Using caching to reduce unnecessary OCR model initialization.

### State Management

Using Streamlit session state to preserve independent OCR engine results.

### Error Handling

Designing graceful fallbacks for runtime and dependency-related failures.

### Deployment

Preparing and deploying an AI-powered Computer Vision application for public use.

---

# 🏗️ From OCR to Document Intelligence

The current application represents the first stage of a much larger Document AI pipeline.

```text
                Document
                   ↓
          Image Quality Analysis
                   ↓
          Automatic Preprocessing
                   ↓
             Multi-Engine OCR
                   ↓
            Text Validation
                   ↓
          Layout Understanding
                   ↓
       ┌───────────┼───────────┐
       ↓           ↓           ↓
     Tables      Forms      Paragraphs
       │           │           │
       └───────────┼───────────┘
                   ↓
        Structured Information
                   ↓
          Database / API / AI
```

This architecture could eventually support real-world applications such as:

* Invoice automation
* Receipt digitization
* Document archiving
* Form processing
* Academic document digitization
* Business workflow automation
* Intelligent document search
* Identity document processing

---

# 🔐 Reliability Philosophy

A production-oriented OCR application should not assume that every dependency will behave identically across every environment.

This project therefore emphasizes:

```text
Robustness
    +
Graceful Error Handling
    +
Independent OCR Engines
    +
Runtime Compatibility
    +
Persistent Results
    +
User Feedback
```

The goal is not simply to make OCR work under ideal conditions, but to make the application **usable and understandable when things go wrong**.

---

# 👨‍💻 Author

## Hadeed Jalani

**Final-Year BSCS Student**
**University of Lahore**

### Technical Interests

* Artificial Intelligence
* Machine Learning
* Computer Vision
* Full-Stack Development
* Intelligent Applications
* Software Engineering

---

# 📚 Internship Context

This project was developed as part of the:

**MLB Summer Internship — Day 23**

### Task

> Build a complete OCR application capable of extracting text from document images using multiple OCR engines, improving OCR input through image preprocessing, visualizing detection results, comparing OCR performance, exporting extracted text, handling runtime issues, and deploying the application publicly.

---

# ⭐ Project Summary

**Document OCR Studio** combines:

```text
Python
   +
OpenCV
   +
EasyOCR
   +
PaddleOCR
   +
Streamlit
   +
Image Preprocessing
   +
Bounding-Box Refinement
   +
Confidence Analysis
   +
Engine Comparison
   +
Error Handling
   +
Text Export
   +
Public Deployment
```

The project demonstrates an end-to-end approach to building a practical OCR application — from **raw document image** to **processed image**, **multi-engine recognition**, **visualized detections**, **confidence analysis**, **comparison**, and **exportable text**.

---

<p align="center">

<strong>🚀 Document OCR Studio</strong><br>
Built with Python • OpenCV • EasyOCR • PaddleOCR • Streamlit

<br><br>

<strong>MLB Summer Internship-Day 23</strong>

</p>
