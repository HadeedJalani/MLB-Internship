<h1 align="center">🏗️ Construction Equipment Detection</h1>
<h3 align="center">Custom YOLO11 Object Detection</h3>

<p align="center">
  <strong>A custom-trained YOLO11 model for detecting construction equipment in images and videos.</strong>
</p>

<p align="center">
  Built using <strong>Ultralytics YOLO11</strong>, trained on a large-scale construction equipment
  dataset using Google Colab GPU, and deployed through an interactive Streamlit application.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/mAP50-94.4%25-brightgreen" />
  <img src="https://img.shields.io/badge/mAP50--95-80.5%25-blue" />
  <img src="https://img.shields.io/badge/model-YOLO11s-orange" />
  <img src="https://img.shields.io/badge/deployed%20with-Streamlit-red" />
</p>

<p align="center">
  <a href="https://github.com/HadeedJalani/MLB-Internship/tree/main/Day28"><strong>💻 GitHub Repository</strong></a> •
  <a href="https://mlb-internship-day28.streamlit.app/"><strong>🌐 Live App</strong></a>
</p>

---

## 📌 Project Overview

This project implements a complete custom object detection pipeline for recognizing construction and heavy equipment.

Unlike pre-trained object detection systems that are limited to generic COCO classes, this project trains a custom YOLO model specifically for construction equipment detection.

The complete workflow includes:

- 📦 Dataset acquisition from Roboflow Universe
- 🔍 Dataset exploration and validation
- 🏷️ YOLO annotation format handling
- 🧠 Custom YOLO11 model training
- 📊 Model evaluation and performance analysis
- 🖼️ Inference on test images
- 🎥 Object detection on videos
- 💾 Prediction result generation
- 🌐 Interactive Streamlit deployment

---

## 🎯 Project Objective

The objective of this project is to build a reliable **Construction Equipment Detection System** capable of detecting multiple categories of heavy machinery from images and videos.

The system performs the following pipeline:

```
Input Image / Video
        │
        ▼
Custom YOLO11 Model
        │
        ▼
Object Detection
        │
        ├── Bounding Boxes
        ├── Class Labels
        └── Confidence Scores
        │
        ▼
Annotated Prediction
        │
        ▼
Downloadable Output
```

---

## 🚀 Key Features

| Feature | Description |
|---|---|
| 🔎 **Custom Object Detection** | Detects construction equipment using a custom-trained YOLO11 model |
| 🖼️ **Image Detection** | Upload images and receive annotated predictions with bounding boxes, class labels, and confidence scores |
| 🎥 **Video Detection** | Supports object detection on uploaded videos with frame-by-frame inference |
| 📊 **Detection Analytics** | Displays total detected objects, detected classes, confidence scores, and processing results |
| ⚙️ **Adjustable Confidence Threshold** | Users can control the minimum confidence required for predictions |
| 💾 **Download Results** | Annotated images and processed videos can be downloaded directly from the app |
| ☁️ **Streamlit Deployment** | The trained custom model is integrated into a deployment-ready Streamlit application |

---

## 🧠 What is Object Detection?

Object Detection is a Computer Vision task that identifies what objects are present in an image and where they are located.

Unlike image classification, which assigns a single label to an entire image, object detection provides:

- Object class
- Bounding box coordinates
- Confidence score

**Example:**

```
Image
 │
 ├── Excavator     → 96%
 ├── Dump Truck    → 91%
 └── Mobile Crane  → 87%
```

Each detected object receives a bounding box describing its location.

### 🆚 Image Classification vs Object Detection vs Segmentation

| Task | Output |
|---|---|
| **Image Classification** | Predicts one or more labels for the entire image |
| **Object Detection** | Detects objects and draws bounding boxes |
| **Image Segmentation** | Classifies individual pixels belonging to objects |

```
Image Classification  →  "This is a construction site"
Object Detection       →  "Excavator here" / "Dump truck here"
Image Segmentation     →  Exact pixels belonging to each object
```

---

## ⚡ Why YOLO?

**YOLO** stands for **You Only Look Once**.

YOLO is a real-time object detection architecture that predicts object locations and classes in a single forward pass through the neural network.

Unlike traditional object detection pipelines that separate region proposal and classification, YOLO performs detection efficiently in one unified architecture.

**Advantages:**
- ⚡ Fast inference
- 🎯 High detection accuracy
- 📦 Lightweight model variants
- 🎥 Suitable for real-time video processing
- 🚀 Easy deployment

---

## 🤖 Model Used

This project uses **YOLO11s**, initialized from pre-trained weights:

```python
model = YOLO("yolo11s.pt")
```

YOLO11s provides a strong balance between detection accuracy, model size, training efficiency, and inference speed.

### Model Specifications

| Property | Value |
|---|---|
| Model | YOLO11s |
| Framework | Ultralytics |
| Parameters | ~9.4 Million |
| GFLOPs | ~21.4 |
| Input Size | 640 × 640 |
| Training Hardware | Tesla T4 GPU |

---

## 📂 Project Structure

```
Day28/
│
├── training/
│   └── Day28_YOLO_Training.ipynb
│
├── models/
│   └── best.pt
│
├── scripts/
│   ├── evaluate_model.py
│   └── test_inference.py
│
├── sample_inputs/
│   ├── images/
│   └── videos/
│
├── outputs/
│   ├── predictions/
│   │   ├── images/
│   │   └── videos/
│   └── evaluation/
│       ├── confusion_matrix.png
│       ├── confusion_matrix_normalized.png
│       ├── F1_curve.png
│       ├── PR_curve.png
│       ├── P_curve.png
│       ├── R_curve.png
│       └── results.png
│
├── app.py
├── requirements.txt
├── runtime.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset

### Dataset Source

**Construction Equipment Dataset**
Source: [Roboflow Universe — Equipment Dataset by SODA1](https://universe.roboflow.com/soda1/equipment-wmk6r)

### Why This Dataset?

The dataset was selected because it contains approximately 10,000 labeled images covering multiple real-world construction and heavy-equipment categories.

It provides:
- Diverse environments
- Different camera angles
- Multiple object scales
- Real-world backgrounds
- Multiple equipment categories

This makes it suitable for training a robust custom object detector.

### 🏷️ Dataset Classes

The model detects the following **10 construction equipment classes**:

1. grader
2. backhoe_loader
3. compactor
4. dozer
5. concrete_mixer_truck
6. excavator
7. wheel_loader
8. mobile_crane
9. dump_truck
10. tower_crane

### 📁 YOLO Dataset Structure

The dataset follows the standard YOLO detection structure:

```
dataset/
│
├── train/
│   ├── images/
│   └── labels/
│
├── valid/
│   ├── images/
│   └── labels/
│
├── test/
│   ├── images/
│   └── labels/
│
└── data.yaml
```

Each image has a corresponding annotation file.

### 📝 YOLO Annotation Format

YOLO annotations follow the format:

```
class_id x_center y_center width height
```

**Example:**
```
5 0.512 0.483 0.324 0.418
```

| Value | Meaning |
|---|---|
| `class_id` | Object class index |
| `x_center` | Normalized bounding box center X |
| `y_center` | Normalized bounding box center Y |
| `width` | Normalized bounding box width |
| `height` | Normalized bounding box height |

All coordinates are normalized between `0` and `1`.

---

## ⚙️ Training Configuration

The model was trained using Google Colab with GPU acceleration.

| Setting | Configuration |
|---|---|
| Base Model | YOLO11s |
| Pre-trained Weights | `yolo11s.pt` |
| Epochs | 100 |
| Early Stopping Patience | 20 |
| Batch Size | 16 |
| Image Size | 640 |
| Optimizer | Auto |
| Learning Rate Schedule | Cosine |
| Hardware | Tesla T4 GPU |
| CUDA | 12.8 |

### 🔄 Data Augmentation

To improve generalization and reduce overfitting, multiple augmentation strategies were used during training:

- Mosaic augmentation
- MixUp augmentation
- HSV color jitter
- Rotation
- Translation
- Scaling
- Shearing
- Horizontal flipping

These augmentations help the model generalize across different camera angles, object orientations, lighting conditions, background environments, and object scales.

---

## 📈 Final Model Performance

The final validation set contained **1,702 images** / **2,609 instances**.

### Overall Metrics

| Metric | Score |
|---|---|
| 🟢 mAP@50 | **94.39%** |
| 🔵 mAP@50-95 | **80.50%** |
| 🎯 Precision | **94.06%** |
| 🔍 Recall | **89.34%** |

### 🏆 Performance Target

**Required:** mAP@50 ≥ 80%
**Achieved:** mAP@50 = 94.39%

✅ **Performance target successfully exceeded by approximately 14.4 percentage points.**

### 📊 Per-Class Performance

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| grader | 199 | 204 | 0.967 | 0.975 | 0.992 | 0.913 |
| backhoe_loader | 219 | 226 | 0.991 | 0.962 | 0.990 | 0.891 |
| compactor | 178 | 191 | 0.962 | 0.937 | 0.974 | 0.884 |
| dozer | 185 | 200 | 0.944 | 0.910 | 0.974 | 0.825 |
| concrete_mixer_truck | 138 | 157 | 0.953 | 0.911 | 0.967 | 0.832 |
| excavator | 382 | 469 | 0.955 | 0.911 | 0.964 | 0.824 |
| wheel_loader | 306 | 327 | 0.936 | 0.899 | 0.950 | 0.824 |
| mobile_crane | 182 | 205 | 0.939 | 0.868 | 0.912 | 0.743 |
| dump_truck | 332 | 557 | 0.882 | 0.847 | 0.907 | 0.686 |
| tower_crane | 52 | 73 | 0.877 | 0.712 | 0.810 | 0.627 |
| **Overall** | **1702** | **2609** | **0.941** | **0.893** | **0.944** | **0.805** |

### ⚡ Inference Performance

Measured during testing on a Tesla T4 GPU:

| Scenario | Speed |
|---|---|
| Validation Batch Inference | ~6.7 ms/image |
| Single Image Inference | ~11.8 ms/image |
| Input Resolution | 640 × 640 |

The model provides a strong balance between detection performance and inference efficiency.

---

## 🔍 Model Evaluation

The trained model generates multiple evaluation artifacts.

**Available Evaluation Outputs:**
```
outputs/evaluation/
│
├── confusion_matrix.png
├── confusion_matrix_normalized.png
├── F1_curve.png
├── PR_curve.png
├── P_curve.png
├── R_curve.png
└── results.png
```

These visualizations help analyze classification errors, precision trends, recall trends, F1 score, the precision-recall relationship, and training/validation loss.

---

## 🧪 Test Image Inference

The trained model was tested on multiple unseen images.

```
Test Image
    ↓
YOLO11s Custom Model
    ↓
Object Detection
    ↓
Bounding Boxes + Class Labels + Confidence Scores
    ↓
Annotated Prediction
```

Predictions are saved inside `outputs/predictions/images/`.

## 🎥 Video Inference

The project also supports object detection on video.

```
Input Video
      ↓
Frame Extraction
      ↓
YOLO Inference
      ↓
Object Detection
      ↓
Bounding Box Rendering
      ↓
Annotated Frames
      ↓
Output Video
```

Output videos are saved inside `outputs/predictions/videos/`.

---

## 🌐 Streamlit Application

The project includes an interactive Streamlit application that allows users to perform inference using the trained custom model.

### Application Features

- 🖼️ **Image Upload** — JPG, JPEG, PNG, WEBP
- 🎥 **Video Upload** — supported video files for frame-by-frame object detection
- 🎯 **Detection Controls** — adjustable confidence threshold, image or video inference
- 📊 **Detection Results** — original input, processed output, bounding boxes, class labels, confidence scores, detection statistics
- ⬇️ **Download Support** — annotated images and processed detection videos

---

## 🚀 Running Locally

**1. Clone the Repository**

```bash
git clone https://github.com/HadeedJalani/MLB-Internship.git
cd MLB-Internship/Day28
```

**2. Install Dependencies**

```bash
pip install -r requirements.txt
```

**3. Verify Model Placement**

Make sure the trained model exists at:

```
models/best.pt
```

**4. Launch Streamlit**

```bash
streamlit run app.py
```

Streamlit will provide a local URL similar to `http://localhost:8501`.

### 🖥️ Running Evaluation

The evaluation script allows the trained model to be evaluated again:

```bash
python scripts/evaluate_model.py \
    --weights models/best.pt \
    --data path/to/data.yaml
```

### 🔎 Running Batch Inference

To perform inference on multiple images:

```bash
python scripts/test_inference.py \
    --weights models/best.pt \
    --source sample_inputs/images \
    --conf 0.25
```

Predictions will be saved automatically.

---

## 🧩 Challenges Faced & Solutions

### 1️⃣ Class Imbalance

The `tower_crane` class contained significantly fewer training examples:

```
tower_crane instances: 73
dump_truck instances:  557
```

**Impact:** the class achieved mAP@50 = 81.0%, Recall = 71.2%.

**Future Improvement:** collecting more tower crane examples, targeted oversampling, class-aware augmentation, additional training iterations.

### 2️⃣ Dump Truck Localization

The `dump_truck` class showed a noticeable gap between:

```
mAP@50      → 90.7%
mAP@50-95   → 68.6%
```

This suggests the model successfully detects dump trucks but has more difficulty producing highly precise bounding box localization across stricter IoU thresholds.

**Possible Cause:** dump trucks appear in the dataset with different scales, different camera angles, partial occlusions, and diverse environments.

### 3️⃣ Dataset Download Reliability

During development, the Roboflow API occasionally returned an incomplete or empty dataset directory.

**Solution:**

```
Download Dataset → Check data.yaml → If Missing → Retry Download → Continue Training
```

This improved the reproducibility of the training pipeline.

---

## 🔧 Future Improvements

- [ ] Train YOLO11m for higher accuracy
- [ ] Collect additional tower crane samples
- [ ] Apply class balancing techniques
- [ ] Hyperparameter optimization
- [ ] Experiment with larger image resolutions
- [ ] Add object tracking for videos
- [ ] Real-time webcam detection
- [ ] Detection analytics dashboard
- [ ] Batch image inference
- [ ] Cloud-based model hosting

---

## ☁️ Deployment

The application is deployed using **Streamlit Community Cloud**.

🔗 **Live Application:** [mlb-internship-day28.streamlit.app](https://mlb-internship-day28.streamlit.app/)

---

## 📎 Submission Links

| Resource | Link |
|---|---|
| 💻 GitHub Repository | [View Repository](https://github.com/HadeedJalani/MLB-Internship/tree/main/Day28) |
| 🌐 Streamlit Application | [Open App](https://mlb-internship-day28.streamlit.app/) |
| 🎥 Demo Video | `[Add link]` |
| 🎬 Model Output Video | `[Add link]` |

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Ultralytics | YOLO model training and inference |
| YOLO11 | Object detection architecture |
| PyTorch | Deep learning backend |
| OpenCV | Image and video processing |
| Streamlit | Interactive web application |
| Google Colab | GPU-based model training |
| Roboflow Universe | Dataset source |

---

## 🎓 Key Learnings

Through this project, hands-on experience was gained with:

- Preparing datasets for YOLO
- Understanding YOLO annotation format
- Working with `data.yaml`
- Training custom object detection models
- Transfer learning with pre-trained weights
- Monitoring training performance
- Evaluating mAP metrics
- Analyzing precision and recall
- Investigating class imbalance
- Running inference on images and videos
- Deploying a trained model with Streamlit

---

## 📌 Final Results

```
Model        : YOLO11s
Classes      : 10 Construction Equipment Categories
Epochs       : 100
Image Size   : 640 × 640

Precision    : 94.06%
Recall       : 89.34%

mAP@50       : 94.39% 🟢
mAP@50-95    : 80.50% 🔵

Target       : mAP@50 ≥ 80%
Status       : TARGET EXCEEDED ✅
```

---

## 👨‍💻 Author

**Hadeed Jalani**
Final-Year Computer Science Student, University of Lahore

Focused on: **Artificial Intelligence • Machine Learning • Computer Vision • Full-Stack Development**

<p align="center">
  <strong>Built as part of the MLBench Summer Internship — Custom Object Detection Journey 🚀</strong>
</p>

<p align="center">
  ⭐ If you found this project useful, consider starring the repository.
</p>
