# 👕 Fashion MNIST Image Classifier using Convolutional Neural Networks (CNN)

## MLBench Summer Internship - Day 13

This project demonstrates the implementation of a Convolutional Neural Network (CNN) for image classification using the Fashion MNIST dataset provided by TensorFlow/Keras.

The application trains a CNN to recognize clothing items and provides an interactive Streamlit web application where users can upload clothing images for prediction.

---

# 📌 What is a Convolutional Neural Network (CNN)?

A Convolutional Neural Network (CNN) is a specialized Deep Learning architecture designed for image processing tasks.

Unlike Artificial Neural Networks (ANNs), CNNs automatically learn important visual features such as edges, textures, patterns, and object shapes using convolution filters.

CNNs are widely used in:

- Image Classification
- Face Recognition
- Medical Image Analysis
- Autonomous Vehicles
- Object Detection
- Security Systems

---

# 📌 Why CNNs are Better than ANNs for Images

Artificial Neural Networks treat every pixel as an independent feature, resulting in a very large number of parameters.

CNNs solve this problem by:

- Learning local image features
- Sharing filter weights
- Preserving spatial information
- Reducing computational complexity
- Achieving higher accuracy on image data

Because of these advantages, CNNs outperform traditional ANNs in computer vision tasks.

---

# 📌 CNN Architecture

The implemented CNN model consists of:

Input Image (28×28×1)

↓

Conv2D (32 Filters, ReLU)

↓

MaxPooling2D

↓

Conv2D (64 Filters, ReLU)

↓

MaxPooling2D

↓

Flatten Layer

↓

Dense Layer (128 Neurons, ReLU)

↓

Dropout Layer

↓

Output Layer (10 Classes, Softmax)

---

# 📌 Dataset

Fashion MNIST

- 70,000 grayscale images
- 60,000 training images
- 10,000 testing images
- 10 clothing categories
- Image size: 28 × 28 pixels

Classes:

- T-shirt/Top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle Boot

---

# 📌 Features

- Load Fashion MNIST dataset
- Dataset preprocessing
- Image normalization
- CNN model training
- Model evaluation
- Prediction on unseen images
- Confusion Matrix
- Training Accuracy graph
- Validation Accuracy graph
- Correct Prediction Visualization
- Incorrect Prediction Visualization
- Interactive Streamlit Web Application

---

# 📌 Model Performance

Training Accuracy:
> (Update after training)

Testing Accuracy:
> (Example: 91–94%)

Loss:
> (Update after training)

---

# 📌 Challenges Faced

- Installing TensorFlow on Python 3.14 (resolved by using Python 3.11)
- Handling image preprocessing
- Improving prediction accuracy
- Displaying prediction confidence
- Deploying TensorFlow applications on Streamlit Cloud

---

# 📌 Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Pandas
- Scikit-learn
- Pillow
- Streamlit

---

# 📂 Project Structure

Day13/

├── cnn_practice.py

├── fashion_mnist_cnn.py

├── streamlit_app.py

├── cnn_model.keras

├── confusion_matrix.png

├── training_accuracy.png

├── training_loss.png

├── correct_predictions.png

├── incorrect_predictions.png

├── requirements.txt

└── README.md

---

# 🚀 How to Run

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Train Model

```bash
python fashion_mnist_cnn.py
```

## Launch Streamlit App

```bash
streamlit run streamlit_app.py
```

---

# 📷 Outputs

- CNN Model Summary
- Training Accuracy
- Validation Accuracy
- Loss Curve
- Confusion Matrix
- Correct Predictions
- Incorrect Predictions
- Streamlit Image Classifier

---

# 👨‍💻 Developed By

**Hadeed Jalani**

MLBench Summer Internship

Day 13 — Convolutional Neural Networks (CNN)