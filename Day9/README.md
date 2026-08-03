# 🌸 Iris Flower Classification System

## MLBench Summer Internship – Day 9

A complete Machine Learning Classification project that predicts the species of an Iris flower using **Logistic Regression** and compares its performance with a **Decision Tree Classifier**.

This project demonstrates the complete machine learning workflow, including dataset exploration, model training, evaluation, prediction, visualization, and deployment using **Streamlit**.

---

# 📌 Project Objectives

- Understand Classification problems
- Learn Logistic Regression
- Compare Logistic Regression with Decision Tree
- Evaluate model performance using different metrics
- Build a complete Iris Flower Classification System
- Deploy the trained model using Streamlit

---

# 📂 Dataset

**Dataset Used:** Iris Dataset

Source:
Scikit-Learn Built-in Dataset

Dataset Information:

- Total Samples: 150
- Features: 4
- Classes: 3

Target Classes:

- Setosa
- Versicolor
- Virginica

Features:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

---

# 📁 Project Structure

```text
Day9/
│
├── classification_practice.py
├── model_evaluation.py
├── iris_classification_system.py
├── streamlit_app.py
├── logistic_regression_model.pkl
├── logistic_regression_confusion_matrix.png
├── requirements.txt
└── README.md
```

---

# 📚 Concepts Covered

## Model Evaluation

- Training Accuracy
- Testing Accuracy
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

---

## Classification

- Classification
- Logistic Regression
- Decision Tree
- Model Prediction
- Probability Prediction

---

# 📈 Machine Learning Workflow

```
Load Dataset
      │
      ▼
Explore Dataset
      │
      ▼
Train-Test Split
      │
      ▼
Train Logistic Regression
      │
      ▼
Train Decision Tree
      │
      ▼
Evaluate Models
      │
      ▼
Generate Confusion Matrix
      │
      ▼
Compare Models
      │
      ▼
Predict Flower Species
      │
      ▼
Deploy with Streamlit
```

---

# 🤖 Algorithms Used

## Logistic Regression

Used as the primary classification model.

Advantages:

- Fast
- Simple
- High Accuracy
- Easy to Interpret

---

## Decision Tree

Used for comparison.

Advantages:

- Easy to Understand
- Handles Non-linear Data
- Good Visualization

---

# 📊 Evaluation Metrics

The following evaluation metrics were used:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

---

# 📈 Model Performance

## Logistic Regression

- Training Accuracy: **97.50%**
- Testing Accuracy: **96.67%**
- Precision: **96.97%**
- Recall: **96.67%**
- F1-Score: **96.66%**

---

## Decision Tree

- Training Accuracy: **100.00%**
- Testing Accuracy: **93.33%**
- Precision: **93.33%**
- Recall: **93.33%**
- F1-Score: **93.33%**

---

# 📊 Observations

- Logistic Regression achieved better testing accuracy.
- Decision Tree achieved perfect training accuracy.
- Decision Tree showed signs of overfitting.
- Logistic Regression generalized better on unseen data.
- Logistic Regression is the recommended model for this dataset.

---

# 🌐 Streamlit Application

The application allows users to:

- Enter flower measurements
- Predict Iris species instantly
- View prediction confidence
- View prediction probabilities
- Explore selected feature values
- Learn about the trained model

---

# 💻 Installation

Clone the repository

```bash
git clone https://github.com/HadeedJalani/MLB-Internship.git
```

Move into Day9

```bash
cd Day9
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the project

```bash
python iris_classification_system.py
```

Run the Streamlit application

```bash
streamlit run streamlit_app.py
```

---

# 📷 Output

The project generates:

- Model Evaluation Metrics
- Classification Report
- Confusion Matrix
- Sample Predictions
- Streamlit Web Application

---

# 🎯 Learning Outcomes

After completing this project, I learned:

- Classification problems
- Logistic Regression
- Decision Trees
- Model Evaluation
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Machine Learning Deployment using Streamlit

---

# 🚀 Future Improvements

- Support multiple classification algorithms
- Hyperparameter tuning
- Cross-validation
- Model persistence with versioning
- Feature importance visualization
- Deployment on Hugging Face Spaces
- Support for custom datasets

---

# 🛠️ Technologies Used

- Python
- Scikit-Learn
- NumPy
- Pandas
- Matplotlib
- Streamlit
- Joblib

---

# 👨‍💻 Author

**Hadeed Jalani**

MLBench Summer Internship – Day 9

University of Lahore

BS Computer Science

---

# ⭐ Acknowledgements

- MLBench Summer Internship
- Scikit-Learn
- Streamlit
- Python Community