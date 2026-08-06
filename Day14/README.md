# 🐱🐶 Cats vs Dogs Image Classifier using Transfer Learning

> **MLBench Summer Internship – Day 14**
>
> A Deep Learning image classification system built using **TensorFlow**, **Keras**, and **MobileNetV2** with **Transfer Learning**. The project classifies images as either **Cat** or **Dog** and includes a professional **Streamlit dashboard** for interactive predictions and model visualization.

---

# 📌 Project Overview

This project demonstrates how Transfer Learning can be used to build a high-performance image classifier without training a convolutional neural network from scratch.

Instead of designing a CNN manually, the project leverages **MobileNetV2** pre-trained on **ImageNet**, allowing faster convergence, improved accuracy, and significantly reduced training time.

The model was trained on the **Microsoft Cats vs Dogs Dataset** containing approximately **25,000 images**.

---

# 🚀 Features

- Transfer Learning using MobileNetV2
- Image Classification (Cat vs Dog)
- Data Augmentation
- Fine-Tuning of Pre-trained Layers
- Model Evaluation
- Confusion Matrix
- Classification Report
- Training & Validation Curves
- Prediction Visualization
- Streamlit Interactive Dashboard
- Image Upload & Prediction
- Image Editing Controls
- Confidence Scores
- Prediction History
- Analytics Dashboard

---

# 🧠 Model Architecture

```
Input Image (224 × 224)

        │

Data Augmentation
(Random Flip, Rotation,
Zoom, Contrast,
Brightness)

        │

MobileNetV2
(ImageNet Weights)

        │

Global Average Pooling

        │

Dropout (0.30)

        │

Dense (256 ReLU)

        │

Dropout (0.20)

        │

Dense (2 Softmax)

        │

Prediction
(Cat / Dog)
```

---

# 🛠 Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- NumPy
- Matplotlib
- Scikit-Learn
- Pillow (PIL)
- Plotly
- Streamlit

---

# 📂 Project Structure

```
Day14/

│
├── cats_vs_dogs_classifier.py
├── streamlit_app.py
├── best_model.keras
├── cats_vs_dogs_model.keras
├── requirements.txt
├── README.md
│
├── accuracy_curve.png
├── loss_curve.png
├── confusion_matrix.png
├── sample_images.png
├── correct_predictions.png
├── incorrect_predictions.png
│
├── test_prediction.py
├── verify_dataset_predictions.py
├── remove_corrupted_images.py
├── check_dataset.py
│
└── dataset/
    ├── train/
    ├── validation/
    └── test/
```

---

# 📊 Dataset

Dataset Used:

**Microsoft Cats vs Dogs Dataset**

Structure:

```
dataset/

├── train
│   ├── Cat
│   └── Dog
│
├── validation
│   ├── Cat
│   └── Dog
│
└── test
    ├── Cat
    └── Dog
```

---

# 📈 Training Configuration

| Parameter | Value |
|-----------|-------|
| Image Size | 224 × 224 |
| Batch Size | 32 |
| Initial Epochs | 10 |
| Fine-Tuning Epochs | 5 |
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Loss Function | Sparse Categorical Crossentropy |

---

# 📊 Model Performance

| Metric | Value |
|---------|-------|
| Test Accuracy | **99.22%** |
| Test Loss | **0.0247** |

The model achieved excellent classification performance after fine-tuning the upper layers of MobileNetV2.

---

# 📷 Streamlit Dashboard

The Streamlit application provides an interactive interface where users can:

- Upload images
- Predict Cats vs Dogs
- View confidence scores
- Adjust brightness
- Adjust contrast
- Rotate images
- Flip images
- Convert to grayscale
- Resize images
- Track prediction history
- View analytics dashboard
- Explore model information

---

# ▶️ Installation

Clone the repository

```bash
git clone https://github.com/HadeedJalani/MLB-Internship.git
```

Move to Day 14

```bash
cd MLB-Internship/Day14
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Train Model

```bash
python cats_vs_dogs_classifier.py
```

---

# ▶️ Launch Streamlit App

```bash
streamlit run streamlit_app.py
```

---

# 📌 Learning Outcomes

Through this project, the following concepts were explored:

- Transfer Learning
- Feature Extraction
- Fine-Tuning
- MobileNetV2
- Data Augmentation
- Model Evaluation
- Confusion Matrix
- Classification Report
- Deep Learning Deployment
- Interactive AI Applications using Streamlit

---

# 👨‍💻 Author

**Hadeed Jalani**

---

# ⭐ Acknowledgements

- TensorFlow
- Keras
- MobileNetV2
- Microsoft Cats vs Dogs Dataset
- MLBench Summer Internship
