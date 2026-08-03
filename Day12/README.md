# 👕 Fashion MNIST Classification using Artificial Neural Networks

## MLBench Summer Internship - Day 12

---

## 📌 Project Overview

This project demonstrates a Deep Learning image classification system using TensorFlow and Keras.

The application is trained on the Fashion MNIST dataset to classify grayscale images of clothing items into one of ten categories.

The project also includes a professional Streamlit web application for interactive image classification.

---

# Deep Learning

Deep Learning is a subset of Machine Learning that uses Artificial Neural Networks (ANNs) with multiple layers to automatically learn patterns from data.

Deep Learning is widely used in:

- Image Classification
- Object Detection
- Speech Recognition
- Natural Language Processing
- Medical Imaging

---

# Machine Learning vs Deep Learning

| Machine Learning | Deep Learning |
|-----------------|--------------|
| Requires manual feature engineering | Learns features automatically |
| Performs well on small datasets | Performs better on large datasets |
| Faster training | Requires more computational power |
| Simpler models | Multi-layer neural networks |

---

# Perceptron

A Perceptron is the basic building block of an Artificial Neural Network.

It receives inputs, applies weights, computes a weighted sum, passes it through an activation function, and produces an output.

---

# Activation Functions

The following activation functions were explored:

### ReLU

Used in hidden layers.

Advantages:

- Fast
- Efficient
- Reduces vanishing gradient problem

---

### Sigmoid

Produces outputs between 0 and 1.

Commonly used in binary classification.

---

### Tanh

Produces outputs between -1 and 1.

Zero-centered and often performs better than Sigmoid in hidden layers.

---

### Softmax

Used in the output layer for multi-class classification.

Converts outputs into probabilities.

---

# Dataset

Fashion MNIST

- 70,000 Images
- 28×28 pixels
- 10 Clothing Categories

Training Images:

- 60,000

Testing Images:

- 10,000

---

# Model Architecture

Input Layer

↓

Flatten Layer

↓

Dense (128, ReLU)

↓

Dense (64, ReLU)

↓

Output Layer (10, Softmax)

---

# Model Performance

Training Accuracy:

≈ 90%

Testing Accuracy:

87.62%

---

# Project Features

- Dataset Exploration
- Data Normalization
- Artificial Neural Network
- Model Training
- Model Evaluation
- Sample Predictions
- Accuracy Graph
- Loss Graph
- Interactive Streamlit Application

---

# Technologies Used

- Python
- TensorFlow
- Keras
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Plotly

---

# How to Run

Install dependencies

```bash
pip install -r requirements.txt
```

Run the training script

```bash
py -3.11 fashion_mnist_ann.py
```

Run the Streamlit application

```bash
py -3.11 -m streamlit run streamlit_app.py
```

---

# Project Structure

```
Day12/

│

├── tensorflow_installation.py

├── simple_ann.py

├── activation_functions.py

├── fashion_mnist_ann.py

├── streamlit_app.py

│

├── fashion_ann_model.keras

├── training_accuracy.png

├── training_loss.png

├── sample_predictions.png

├── sample_images.png

│

├── README.md

└── requirements.txt
```

---

# Developer

Hadeed Jalani

MLBench Summer Internship

Day 12