# ✂️ Day 26  Image Segmentation

> **MLB Internship | Computer Vision | Image Processing**

A practical Computer Vision project focused on **image segmentation and thresholding** using OpenCV. The project explores multiple segmentation techniques, compares their behavior across different image conditions, and provides an interactive **Streamlit application** for real-time experimentation.

---

## 🌐 Live Demo

 **Streamlit App:**
https://mlb-internship-day26.streamlit.app/

 **GitHub Repository:**
https://github.com/HadeedJalani/MLB-Internship/tree/main/Day26

---

## 🎯 Project Objective

The objective of Day 26 was to understand how image segmentation can separate the **foreground/object of interest from the background**.

The project implements and compares:

* Binary Thresholding
* Adaptive Thresholding
* Otsu Thresholding
* Foreground/Background Segmentation
* Morphological image processing
* Interactive result visualization
* Downloadable processed images

The application is designed to work with both **documents and simple objects**, particularly under different lighting and background conditions.

---

## 🧠 What Is Image Segmentation?

Image segmentation is the process of dividing an image into meaningful regions based on characteristics such as:

* Pixel intensity
* Color
* Texture
* Spatial information
* Object boundaries

Unlike object detection, which generally identifies an object using a bounding box, segmentation works at the **pixel level**.

### Object Detection

```text
Image
  ↓
Object Detection
  ↓
Bounding Box
```

### Image Segmentation

```text
Image
  ↓
Segmentation
  ↓
Pixel-level foreground/background mask
```

This makes segmentation useful for applications such as:

* 📄 Document processing
* 🚗 Autonomous vehicles
* 🏥 Medical imaging
* 🌱 Agriculture
* 🛒 Product image processing
* 🖼️ Background removal
* 🔍 Industrial inspection

---

# 🔬 Segmentation Methods

## 1. Binary Thresholding

Binary thresholding uses a fixed threshold value to divide grayscale pixels into two groups.

Pixels above the threshold become white, while pixels below it become black.

### Concept

```text
Pixel < Threshold  →  0
Pixel ≥ Threshold  →  255
```

### Best suited for

* Clean documents
* High-contrast images
* Uniform lighting
* Simple foreground/background separation

### Limitation

A single global threshold can perform poorly when an image contains shadows or uneven illumination.

---

## 2. Adaptive Thresholding

Adaptive thresholding calculates a threshold independently for local regions of an image.

This makes it more suitable for images where lighting is not uniform.

### Best suited for

* Documents photographed with a phone
* Uneven illumination
* Shadows
* Receipts
* Paper with varying brightness

### Advantage

Instead of using one threshold for the entire image, different areas can receive different thresholds.

---

## 3. Otsu Thresholding

Otsu thresholding automatically determines an appropriate global threshold from the image histogram.

Instead of manually selecting a threshold, OpenCV determines the threshold that provides good separation between foreground and background intensity distributions.

### Best suited for

* Images with clearly separated foreground/background intensities
* Automatically determining a threshold
* General-purpose grayscale segmentation

### Limitation

Otsu is still a **global thresholding method**, so strong shadows or highly uneven lighting can reduce its effectiveness.

---

## 4. Foreground Segmentation

The project also includes a simple foreground/background segmentation pipeline.

The process includes:

```text
Input Image
     ↓
Grayscale Conversion
     ↓
Gaussian Blur
     ↓
Otsu Thresholding
     ↓
Morphological Opening
     ↓
Morphological Closing
     ↓
Foreground Mask
     ↓
Segmented Object
```

Morphological operations help remove small noise and close gaps in the segmentation mask.

---

# 🧹 Image Processing

Several OpenCV operations are used throughout the project.

### Grayscale Conversion

Converts the original BGR image into a single-channel grayscale image.

### Gaussian Blur

Reduces image noise before segmentation.

### Morphological Opening

Helps remove small unwanted foreground regions.

### Morphological Closing

Helps fill small holes and connect nearby regions.

### Binary Mask

Represents the segmented foreground using pixel values of `0` and `255`.

---

# 📊 Dataset

A dataset of **15 images** was used to evaluate the segmentation methods.

The dataset includes different image conditions:

| Image Type            | Purpose                           |
| --------------------- | --------------------------------- |
| 📄 Documents          | Test text/document segmentation   |
| 📦 Simple objects     | Test foreground extraction        |
| 💡 Uneven lighting    | Test adaptive methods             |
| 🌑 Shadows            | Test segmentation robustness      |
| 🖼️ Plain backgrounds | Test object/background separation |

The goal was not simply to process clean images, but to observe how each segmentation method behaves under realistic conditions.

---

# 🧪 Method Comparison

| Method     | Threshold Type    | Lighting Robustness | Best Use Case                   |
| ---------- | ----------------- | ------------------: | ------------------------------- |
| Binary     | Global / Fixed    |                  ⭐⭐ | Clean high-contrast images      |
| Adaptive   | Local             |               ⭐⭐⭐⭐⭐ | Uneven lighting & documents     |
| Otsu       | Automatic Global  |                ⭐⭐⭐⭐ | Naturally separated intensities |
| Foreground | Otsu + Morphology |                ⭐⭐⭐⭐ | Simple foreground extraction    |

### General observations

**Binary Thresholding** performs well when the foreground and background have clearly different intensity values.

**Adaptive Thresholding** is generally more reliable for photographed documents where shadows and uneven lighting are present.

**Otsu Thresholding** provides a convenient automatic threshold without requiring manual parameter selection.

**Foreground Segmentation** improves basic thresholding by adding morphological cleanup, making the resulting mask cleaner for simple objects.

---

# 🖥️ Streamlit Application

The project includes an interactive Streamlit interface.

Users can:

1. 📤 Upload an image
2. ⚙️ Select a segmentation method
3. 🚀 Run the segmentation
4. 🖼️ Compare the original and segmented image
5. ⏱️ View processing time
6. 📥 Download the processed result

### Supported methods

```text
Binary Thresholding
Adaptive Thresholding
Otsu Thresholding
Foreground Segmentation
```

---

# 🏗️ Project Architecture

```text
Day26/
│
├── app.py
│
├── segmentation.py
│
├── requirements.txt
│
├── README.md
│
├── sample_images/
│   ├── documents/
│   ├── objects/
│   ├── shadows/
│   └── uneven_lighting/
│
└── outputs/
    ├── binary/
    ├── adaptive/
    ├── otsu/
    └── foreground/
```

### `app.py`

Contains the Streamlit user interface.

Responsibilities include:

* Image uploading
* Method selection
* Running segmentation
* Result visualization
* Processing-time measurement
* Download functionality

### `segmentation.py`

Contains the core Computer Vision functionality.

Responsibilities include:

* Image reading
* Grayscale conversion
* Binary thresholding
* Adaptive thresholding
* Otsu thresholding
* Foreground segmentation
* Morphological processing
* Mask application
* Image saving

### `requirements.txt`

Contains the lightweight CPU-compatible dependencies required to run the Streamlit application.

---

# ⚙️ Technologies Used

| Technology   | Purpose                         |
| ------------ | ------------------------------- |
| 🐍 Python    | Application development         |
| 👁️ OpenCV   | Image processing & segmentation |
| 🔢 NumPy     | Numerical image operations      |
| 🖼️ Pillow   | Image loading and conversion    |
| 🎨 Streamlit | Interactive web application     |

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/HadeedJalani/MLB-Internship.git
```

Navigate to Day 26:

```bash
cd MLB-Internship/Day26
```

Create a virtual environment:

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Locally

Start the Streamlit application:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal, typically:

```text
http://localhost:8501
```

---

# ☁️ Streamlit Deployment

The application is deployed using **Streamlit Community Cloud**.

### Deployment configuration

```text
Repository:
HadeedJalani/MLB-Internship

Branch:
main

Main file:
Day26/app.py
```

### Live application

🚀 https://mlb-internship-day26.streamlit.app/

The application uses lightweight CPU-compatible dependencies, making it suitable for cloud deployment without requiring GPU hardware.

---

# 📥 Output

After processing an image, the application displays:

```text
Original Image
       ↓
Selected Segmentation Method
       ↓
Segmented Image
       ↓
Processing Time
       ↓
Download Result
```

The segmented image can be downloaded directly from the Streamlit interface as a PNG file.

---

# ⚠️ Challenges

Several challenges were considered during development.

### Uneven Lighting

A fixed threshold may produce inconsistent segmentation when different parts of an image have different brightness levels.

**Solution:** Adaptive thresholding provides locally calculated thresholds.

### Shadows

Shadows can sometimes be interpreted as part of the foreground.

**Solution:** Adaptive thresholding and morphological processing can improve the result, depending on the image.

### Noise

Small unwanted regions may appear in binary masks.

**Solution:** Morphological opening and closing are used to clean the foreground mask.

### Different Image Types

No single segmentation method performs perfectly on every image.

**Solution:** The application provides multiple methods so users can select the technique that best matches their input.

---

# 📈 Key Learnings

Through this project, I learned:

* What image segmentation is
* Difference between segmentation and object detection
* How binary thresholding works
* How adaptive thresholding handles uneven illumination
* How Otsu automatically selects a threshold
* How morphological operations improve masks
* How foreground/background segmentation works
* How to compare different segmentation approaches
* How to build an interactive Computer Vision application
* How to deploy an OpenCV-based application on Streamlit Cloud

---

# 🚀 Future Improvements

Possible improvements include:

* 🎯 Watershed-based segmentation
* ✂️ GrabCut foreground extraction
* 🎨 Color-based segmentation
* 🧠 Deep-learning segmentation models
* 📄 Automatic document boundary detection
* 📐 Perspective correction
* 🧹 Advanced shadow removal
* 📊 Automated quality comparison between methods
* 📁 Batch processing for multiple images
* 🖼️ Side-by-side comparison of all segmentation methods

---

# 📸 Application Workflow

```text
                Upload Image
                     │
                     ▼
             ┌───────────────┐
             │  Select Method │
             └───────┬───────┘
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       Binary     Adaptive     Otsu
          │          │          │
          └──────────┼──────────┘
                     ▼
           Foreground Processing
                     │
                     ▼
             Segmented Output
                     │
             ┌───────┴───────┐
             ▼               ▼
        View Result      Download PNG
```

---

# 📋 Day 26 Deliverables

| Deliverable               | Status |
| ------------------------- | ------ |
| Segmentation scripts      | ✅      |
| Binary Thresholding       | ✅      |
| Adaptive Thresholding     | ✅      |
| Otsu Thresholding         | ✅      |
| Foreground Segmentation   | ✅      |
| 15-image dataset          | ✅      |
| Output image generation   | ✅      |
| Streamlit application     | ✅      |
| Downloadable output       | ✅      |
| CPU-compatible deployment | ✅      |
| GitHub repository         | ✅      |
| Streamlit deployment      | ✅      |

---

## 🔗 Project Links

### GitHub

https://github.com/HadeedJalani/MLB-Internship/tree/main/Day26

### Live Streamlit Application

https://mlb-internship-day26.streamlit.app/

---

## 👨‍💻 Project

**MLB Internship — Day 26**

**Topic:** Introduction to Image Segmentation
**Framework:** Streamlit
**Computer Vision:** OpenCV
**Language:** Python

---

> **Day 26 completed — from basic thresholding to a deployed interactive image segmentation application. 🚀**
