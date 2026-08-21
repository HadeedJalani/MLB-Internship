# 📄 Document OCR Studio

### Multi-Engine OCR Application with EasyOCR, PaddleOCR & RapidOCR

<p align="center">
  <strong>Extract text from images using multiple OCR engines, preprocess documents, compare results, and visualize detected text directly through an interactive Streamlit interface.</strong>
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge\&logo=opencv\&logoColor=white)
![EasyOCR](https://img.shields.io/badge/EasyOCR-OCR-1F2937?style=for-the-badge)
![PaddleOCR](https://img.shields.io/badge/PaddleOCR-OCR-0A66C2?style=for-the-badge)
![RapidOCR](https://img.shields.io/badge/RapidOCR-OCR-16A34A?style=for-the-badge)

</p>

---

## 🌐 Live Demo

### 🚀 Try Document OCR Studio

**Streamlit Application:**
https://mlb-internship-mhgum6bhfkujukngqbt3jr.streamlit.app/

Upload an image, select the available OCR engines, preprocess the document, and extract text directly from your browser.

---

# 📌 Table of Contents

* [Overview](#-overview)
* [Project Objective](#-project-objective)
* [Key Features](#-key-features)
* [How the System Works](#-how-the-system-works)
* [OCR Engines](#-ocr-engines)
* [Image Preprocessing](#-image-preprocessing)
* [Architecture](#-architecture)
* [Project Structure](#-project-structure)
* [Technology Stack](#-technology-stack)
* [Requirements](#-requirements)
* [Installation](#-installation)
* [Running Locally](#-running-locally)
* [Using the Application](#-using-the-application)
* [OCR Result Processing](#-ocr-result-processing)
* [Engine Comparison](#-engine-comparison)
* [Streamlit Deployment](#-streamlit-deployment)
* [Deployment Configuration](#-deployment-configuration)
* [Troubleshooting](#-troubleshooting)
* [Performance Considerations](#-performance-considerations)
* [Limitations](#-limitations)
* [Future Improvements](#-future-improvements)
* [Learning Outcomes](#-learning-outcomes)
* [Project Status](#-project-status)
* [Author](#-author)

---

# 🔎 Overview

**Document OCR Studio** is a multi-engine Optical Character Recognition (OCR) application developed as part of the **MLBench Internship - Day 24**.

The application is designed to extract machine-readable text from uploaded document images using multiple OCR engines:

* **EasyOCR**
* **PaddleOCR**
* **RapidOCR**

Instead of depending on a single OCR implementation, the project provides an architecture where different OCR engines can be loaded and executed independently.

The application combines OCR with **OpenCV-based image preprocessing**, allowing documents to be enhanced before text extraction.

A Streamlit-based interface provides an interactive workflow for:

1. Uploading an image
2. Previewing the document
3. Applying preprocessing
4. Selecting OCR engines
5. Running OCR
6. Comparing extracted results
7. Visualizing detected text regions
8. Reviewing extracted text

---

# 🎯 Project Objective

The main objective of this project is to build a practical OCR pipeline that demonstrates how different OCR engines can be integrated into a single application.

The project focuses on:

* Understanding OCR pipelines
* Working with multiple OCR frameworks
* Image preprocessing using OpenCV
* Text detection and recognition
* Handling different OCR output formats
* Building modular Python components
* Running OCR engines independently
* Managing optional dependencies
* Improving application reliability
* Creating an interactive Streamlit UI
* Deploying an AI/computer-vision application to Streamlit Cloud

---

# ✨ Key Features

## 📤 Image Upload

Upload document images directly through the Streamlit interface.

Supported image formats include common formats such as:

* PNG
* JPG
* JPEG
* WEBP

---

## 🔍 Multi-Engine OCR

The application integrates three OCR engines:

### EasyOCR

A deep-learning-based OCR library that supports multiple languages and provides both text detection and recognition.

### PaddleOCR

A powerful OCR framework capable of handling document text detection and recognition.

The implementation is isolated so that PaddleOCR compatibility issues do not crash the entire application.

### RapidOCR

A lightweight OCR solution designed for efficient text recognition.

---

## 🖼️ Image Preprocessing

Uploaded images can be processed before OCR to improve recognition quality.

The preprocessing pipeline is designed around OpenCV and can include operations such as:

* Image resizing
* Grayscale conversion
* Noise reduction
* Thresholding
* Contrast enhancement
* Image normalization
* Other document-oriented transformations

Preprocessing is particularly useful for:

* Low-quality scans
* Blurry documents
* Uneven lighting
* Low-contrast text
* Noisy backgrounds

---

## 📦 Modular OCR Architecture

Each OCR engine is implemented as a separate module.

This keeps the project maintainable and makes it easier to:

* Add new OCR engines
* Replace an existing engine
* Debug individual engines
* Compare OCR performance
* Handle engine-specific APIs

---

## ⚡ Lazy Engine Initialization

OCR engines are not unnecessarily initialized when the application starts.

Instead, engines are loaded when they are required.

This helps reduce:

* Initial startup time
* Memory consumption
* Unnecessary model loading
* Deployment failures

---

## 🛡️ Fault Isolation

An important design goal of the application is that one OCR engine should not bring down the entire application.

For example, if PaddleOCR is unavailable because of a dependency or compatibility issue, the application can continue operating with the other available OCR engines.

---

## 📊 OCR Result Visualization

Detected text regions can be visualized on the original image.

The application can display bounding boxes around detected text areas, allowing users to understand where OCR engines identified text.

---

## 🧹 Result Processing

OCR outputs differ between libraries.

The project therefore contains dedicated result-processing utilities that normalize and organize OCR output before presenting it to the user.

---

## 🌐 Interactive Streamlit Interface

The complete OCR pipeline is exposed through an easy-to-use web interface.

Users do not need to interact with Python code directly.

---

# ⚙️ How the System Works

The complete workflow can be summarized as:

```text
                    ┌──────────────────┐
                    │   Upload Image   │
                    └────────┬─────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Image Preprocessing  │
                  │      OpenCV          │
                  └──────────┬───────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        ┌──────────┐   ┌───────────┐   ┌───────────┐
        │ EasyOCR  │   │ PaddleOCR │   │ RapidOCR  │
        └────┬─────┘   └─────┬─────┘   └─────┬─────┘
             │               │               │
             └───────────────┼───────────────┘
                             ▼
                  ┌──────────────────────┐
                  │ Result Normalization │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Text + Bounding Box  │
                  │     Visualization    │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Streamlit Interface  │
                  └──────────────────────┘
```

---

# 🤖 OCR Engines

## 1. EasyOCR

EasyOCR provides a relatively simple interface for extracting text from images.

The project isolates its implementation inside:

```text
ocr/easyocr_engine.py
```

Responsibilities include:

* Initializing the EasyOCR reader
* Running OCR
* Returning detected text
* Returning confidence information
* Returning bounding-box information

---

## 2. PaddleOCR

PaddleOCR is integrated as a separate engine:

```text
ocr/paddleocr_engine.py
```

Because PaddleOCR APIs and package versions can vary, the implementation is designed to handle different API structures where possible.

The application also isolates PaddleOCR failures so that an unavailable PaddleOCR installation does not necessarily prevent other OCR engines from working.

---

## 3. RapidOCR

RapidOCR is implemented in:

```text
ocr/rapidocr_engine.py
```

RapidOCR provides another OCR pipeline that can be used independently of EasyOCR and PaddleOCR.

This makes it possible to compare multiple OCR approaches using the same input document.

---

# 🖼️ Image Preprocessing

OCR performance depends heavily on the quality of the input image.

The project therefore includes an image-processing module:

```text
utils/image_processing.py
```

The preprocessing layer is responsible for preparing images before they are passed to OCR engines.

Typical preprocessing operations include:

### Grayscale Conversion

Converts a color image into grayscale, reducing unnecessary color information.

### Noise Reduction

Reduces unwanted visual noise that could interfere with OCR.

### Thresholding

Separates foreground text from the background.

### Contrast Enhancement

Improves the distinction between text and the document background.

### Resizing

Adjusts image dimensions when necessary to improve OCR processing.

The preprocessing architecture is modular so additional techniques can be added without modifying the OCR engine implementations.

---

# 🏗️ Architecture

The project follows a modular architecture:

```text
Streamlit UI
     │
     ▼
Application Controller
     │
     ├───────────────┐
     │               │
     ▼               ▼
Preprocessing     OCR Engines
     │               │
     │        ┌──────┼──────┐
     │        ▼      ▼      ▼
     │     EasyOCR Paddle RapidOCR
     │        │      │      │
     └────────┼──────┼──────┘
              ▼
       Result Processing
              │
              ▼
      Visualization / UI
```

This separation makes the system easier to maintain and extend.

---

# 📁 Project Structure

```text
Day24/
│
├── app.py
│
├── README.md
├── requirements.txt
├── runtime.txt
│
├── ocr/
│   ├── __init__.py
│   ├── easyocr_engine.py
│   ├── paddleocr_engine.py
│   └── rapidocr_engine.py
│
└── utils/
    ├── __init__.py
    ├── image_processing.py
    └── result_processing.py
```

### `app.py`

Main Streamlit application.

Responsible for:

* UI
* Image upload
* OCR workflow
* Session state
* Engine selection
* OCR execution
* Result presentation
* Visualization

---

### `ocr/`

Contains individual OCR engine implementations.

```text
easyocr_engine.py
paddleocr_engine.py
rapidocr_engine.py
```

Each module encapsulates engine-specific logic.

---

### `utils/`

Contains reusable helper functions.

#### `image_processing.py`

Handles image preprocessing.

#### `result_processing.py`

Handles OCR result normalization, parsing, and formatting.

---

### `requirements.txt`

Contains Python dependencies required by the application.

---

### `runtime.txt`

Specifies the Python runtime used by the deployment environment.

---

# 🛠️ Technology Stack

| Technology      | Purpose                               |
| --------------- | ------------------------------------- |
| Python          | Core programming language             |
| Streamlit       | Interactive web application           |
| OpenCV          | Image processing and preprocessing    |
| EasyOCR         | OCR engine                            |
| PaddleOCR       | OCR engine                            |
| RapidOCR        | OCR engine                            |
| NumPy           | Numerical/image processing operations |
| Pillow          | Image handling                        |
| Git             | Version control                       |
| GitHub          | Source-code hosting                   |
| Streamlit Cloud | Application deployment                |

---

# 📋 Requirements

Before running the application locally, make sure you have:

* Python 3.11 recommended
* pip
* Git
* A working virtual environment
* Internet connection for downloading OCR models

Python 3.11 is recommended because OCR frameworks can have compatibility limitations with newer Python releases.

---

# 🚀 Installation

## 1. Clone the Repository

From your terminal:

```bash
git clone https://github.com/HadeedJalani/MLB-Internship.git
```

Navigate into the repository:

```bash
cd MLB-Internship
```

Then enter the Day24 directory:

```bash
cd Day24
```

---

## 2. Create a Virtual Environment

### Windows

```powershell
python -m venv venv
```

Activate it:

```powershell
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

## 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

Depending on the selected OCR engines, installation may download large machine-learning dependencies and OCR models.

---

# ▶️ Running Locally

From the `Day24` directory:

```bash
streamlit run app.py
```

Streamlit will start a local development server.

The terminal will provide a local address similar to:

```text
http://localhost:8501
```

Open that address in your browser.

---

# 🖥️ Using the Application

## Step 1 - Upload an Image

Upload a document image using the Streamlit file uploader.

---

## Step 2 - Preview the Image

The uploaded document is displayed inside the application.

This allows the user to verify that the correct image has been selected.

---

## Step 3 - Preprocess the Image

Apply available preprocessing techniques to improve the quality of the document before OCR.

---

## Step 4 - Select OCR Engine(s)

Choose the OCR engine you want to use:

```text
EasyOCR
PaddleOCR
RapidOCR
```

Multiple engines can be used for comparison.

---

## Step 5 - Run OCR

The selected engines process the document.

Depending on the engine, the result may include:

* Extracted text
* Confidence score
* Bounding boxes
* Detection information

---

## Step 6- Review Results

The extracted text is presented through the Streamlit interface.

Detected regions can also be visualized on the image.

---

# 🧩 OCR Result Processing

Different OCR libraries return results in different formats.

For example, one engine may return:

```text
[
    (bounding_box, text, confidence)
]
```

while another may return nested detection and recognition structures.

To avoid coupling the Streamlit interface to individual OCR APIs, result processing is handled separately.

The module:

```text
utils/result_processing.py
```

is responsible for transforming engine-specific results into a more consistent internal representation.

Conceptually, the application works with information such as:

```text
Text
Confidence
Bounding Box
Engine
```

This makes the UI independent of the underlying OCR implementation.

---

# 📊 Engine Comparison

One of the major purposes of the project is to demonstrate that OCR engines can behave differently on the same document.

Factors that can influence OCR results include:

* Image resolution
* Font type
* Font size
* Image noise
* Rotation
* Document layout
* Lighting
* Text orientation
* Language
* Background complexity

A practical OCR application should therefore avoid assuming that one engine will always produce the best result.

The multi-engine architecture allows future benchmarking based on:

* Accuracy
* Processing time
* Confidence
* Detection quality
* Resource consumption

---

# ☁️ Streamlit Deployment

The application is deployed using **Streamlit Cloud**.

### Live Application

🚀 **https://mlb-internship-mhgum6bhfkujukngqbt3jr.streamlit.app/**

The deployment uses the Day24 application as the Streamlit entry point.

---

# 🔧 Deployment Configuration

The Day24 deployment includes:

```text
Day24/
├── app.py
├── requirements.txt
└── runtime.txt
```

The project also uses a repository-level:

```text
packages.txt
```

for required Linux system packages.

---

## `runtime.txt`

This file specifies the Python runtime required by the application.

Python 3.11 is used/recommended for compatibility with the OCR ecosystem.

---

## `requirements.txt`

This file defines Python packages required by the application.

The dependency list includes the frameworks used for:

* Streamlit
* OpenCV
* OCR
* Image processing
* Numerical operations

For cloud/server deployment, the OpenCV dependency should use the headless build where GUI functionality is unnecessary:

```text
opencv-python-headless
```

---

## `packages.txt`

Streamlit Cloud runs on a Linux environment.

Some Python packages may require system-level libraries.

For OpenCV-related Linux dependencies, the repository includes:

```text
libgl1
```

in:

```text
packages.txt
```

This addresses system-library requirements such as:

```text
libGL.so.1
```

that may otherwise cause OpenCV import failures in the cloud environment.

---

# 🐛 Troubleshooting

## `ImportError: libGL.so.1`

If you encounter:

```text
ImportError: libGL.so.1: cannot open shared object file
```

make sure the repository root contains:

```text
packages.txt
```

with:

```text
libgl1
```

Also prefer:

```text
opencv-python-headless
```

for Streamlit Cloud deployments.

---

## OCR Engine Not Available

OCR engines may fail to initialize because of:

* Missing dependencies
* Unsupported Python versions
* Model download failures
* Framework version conflicts
* Insufficient resources

The application is designed to isolate engine initialization where possible so that one unavailable engine does not necessarily prevent the application from starting.

---

## PaddleOCR Compatibility

PaddleOCR has had API differences across versions.

The application therefore keeps PaddleOCR implementation isolated inside:

```text
ocr/paddleocr_engine.py
```

This allows PaddleOCR-specific compatibility logic to remain separate from the main application.

---

## Application Takes Time to Start

OCR libraries can download and initialize machine-learning models.

The first application startup can therefore take longer than subsequent operations.

Lazy initialization is used to avoid loading every OCR model unnecessarily.

---

# ⚡ Performance Considerations

OCR is computationally expensive compared with simple image-processing operations.

Performance can be affected by:

* Image resolution
* Number of OCR engines selected
* OCR model size
* CPU availability
* Memory availability
* Number of preprocessing operations

The application uses lazy initialization and modular execution to reduce unnecessary work.

The OCR engines can also be executed independently, allowing the application architecture to support concurrent execution where appropriate.

---

# 🔐 Error Handling

The application attempts to prevent individual OCR engine failures from crashing the entire interface.

For example:

```text
Application
    │
    ├── EasyOCR ──────── Success
    │
    ├── PaddleOCR ────── Failure
    │
    └── RapidOCR ─────── Success
```

Instead of completely terminating the application, the failed engine can be reported while other available engines continue operating.

This approach improves application resilience.

---

# ⚠️ Limitations

Although the application supports multiple OCR engines, OCR quality is not guaranteed for every document.

Potential limitations include:

* Handwritten text
* Extremely low-resolution images
* Highly distorted documents
* Complex tables
* Multi-column layouts
* Heavy image noise
* Unusual fonts
* Strong image rotation
* Poor lighting
* Unsupported languages
* Very large images
* Cloud resource limitations

Different OCR engines may also produce different results for the same image.

---

# 🔮 Future Improvements

Possible future improvements include:

### 📈 OCR Benchmarking

Create a dedicated benchmarking system comparing:

* Accuracy
* Speed
* Confidence
* Memory usage
* Detection quality

---

### 🌍 Expanded Language Support

Allow users to select OCR languages dynamically.

---

### 📑 PDF Support

Extend the application from image-only OCR to PDF documents.

Possible pipeline:

```text
PDF
 ↓
Page Extraction
 ↓
Image Conversion
 ↓
Preprocessing
 ↓
OCR
 ↓
Structured Text
```

---

### 🧾 Structured Document Extraction

Move beyond plain text extraction and identify:

* Headings
* Tables
* Paragraphs
* Dates
* Names
* Addresses
* Key-value pairs

---

### 📊 OCR Analytics

Provide statistics such as:

* Total detected words
* Average confidence
* Processing time
* Number of detected text regions
* Engine comparison metrics

---

### 💾 Export Results

Allow users to download OCR output as:

```text
TXT
CSV
JSON
PDF
DOCX
```

---

### 🧠 Advanced Document Understanding

Integrate NLP/LLM-based processing after OCR to perform tasks such as:

* Document summarization
* Information extraction
* Classification
* Question answering
* Entity extraction

---

# 🎓 Learning Outcomes

Through this project, the following concepts were practiced:

* Optical Character Recognition
* Computer Vision
* Image preprocessing
* OpenCV
* OCR model integration
* EasyOCR
* PaddleOCR
* RapidOCR
* Bounding-box processing
* Confidence-score handling
* Modular Python architecture
* Error handling
* Lazy initialization
* Concurrent processing concepts
* Streamlit application development
* Dependency management
* Linux system dependencies
* Cloud deployment
* Git/GitHub workflow

---

# 🧪 Project Development Workflow

The project followed a modular development process:

```text
1. Understand OCR
        ↓
2. Integrate OCR engines
        ↓
3. Build preprocessing pipeline
        ↓
4. Normalize OCR results
        ↓
5. Build Streamlit interface
        ↓
6. Add error handling
        ↓
7. Optimize engine initialization
        ↓
8. Configure deployment dependencies
        ↓
9. Deploy to Streamlit Cloud
        ↓
10. Test and troubleshoot
```

---

# 📌 Project Status

**Status: ✅ Completed**

The project includes:

* ✅ Multi-engine OCR
* ✅ EasyOCR integration
* ✅ PaddleOCR integration
* ✅ RapidOCR integration
* ✅ Image preprocessing
* ✅ OCR result processing
* ✅ Bounding-box visualization
* ✅ Lazy OCR initialization
* ✅ Error isolation
* ✅ Streamlit interface
* ✅ Cloud deployment
* ✅ Deployment dependency configuration

---

# 🌐 Live Demo

Try the deployed application:

### 🚀 Document OCR Studio

**https://mlb-internship-mhgum6bhfkujukngqbt3jr.streamlit.app/**

---

# 👨‍💻 Author

## Hadeed Jalani

**Final-Year BSCS Student | Full-Stack & AI Development**

This project was developed as part of the **MLBench Internship — Day 24**.

---

<p align="center">

### 📄 Document OCR Studio

**Multi-Engine OCR • Computer Vision • Streamlit • AI**

Made with Python and ❤️ by **Hadeed Jalani**

</p>
