# 🩺 Day 10 - Breast Cancer Prediction System

## MLBench Summer Internship

This project focuses on **Model Evaluation** and **Hyperparameter Tuning** using the **Breast Cancer Wisconsin Diagnostic Dataset** from Scikit-learn.

A Logistic Regression classifier was trained to predict whether a tumor is **Malignant** or **Benign**. The model was evaluated using multiple performance metrics and then improved using **GridSearchCV**.

---

# 📂 Project Structure

```
Day10/
│
├── baseline_model.py
├── breast_cancer_prediction_system.py
├── confusion_matrix_baseline.png
├── confusion_matrix_tuned.png
├── dataset_exploration.py
├── hyperparameter_tuning.py
├── logistic_regression_model.pkl
├── scaler.pkl
├── streamlit_app.py
├── train_model.py
├── requirements.txt
└── README.md
```

---

# 📖 Topics Covered

- Model Evaluation
- Train vs Test Performance
- Underfitting vs Overfitting
- Cross Validation
- Logistic Regression
- Hyperparameter Tuning
- GridSearchCV
- Classification Metrics
- Confusion Matrix
- Streamlit Deployment

---

# 📊 Dataset

Dataset Used:

**Breast Cancer Wisconsin Diagnostic Dataset**

Source:

Scikit-learn (`load_breast_cancer()`)

Dataset Statistics:

- Samples : **569**
- Features : **30**
- Classes : **2**
  - Malignant
  - Benign

---

# 🔍 Dataset Exploration

The dataset exploration script performs:

- Loading the dataset
- Creating a Pandas DataFrame
- Displaying:
  - Head
  - Info
  - Describe
- Checking target distribution

---

# 🤖 Baseline Model

Algorithm:

**Logistic Regression**

The baseline model was trained using an 80-20 train-test split.

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

# ⚙ Hyperparameter Tuning

GridSearchCV was used to search for the best model parameters.

Parameters Tested:

```python
{
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["liblinear", "lbfgs"]
}
```

Cross Validation:

**5-Fold Cross Validation**

---

# 🏆 Best Parameters

Example:

```python
{
    'C': 0.1, 'solver': 'lbfgs'
    
}
```

*(Update this section according to your actual GridSearchCV output.)*

---

# 📈 Model Evaluation

The following metrics were used:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Both the **Baseline** and **Tuned** models were compared to evaluate performance improvements.

---

# 🖥 Streamlit Application

The Streamlit application provides:

- Dataset Upload
- Dataset Preview
- Dataset Statistics
- Live Prediction using Sliders
- Manual Feature Input
- Prediction Confidence
- Probability Distribution
- Baseline Model Metrics
- Tuned Model Metrics
- Best Hyperparameters
- Classification Reports
- Confusion Matrices
- Downloadable Comparison Report

---

# 📚 What I Learned

During this task I learned:

- The importance of evaluating machine learning models using multiple metrics instead of relying only on accuracy.
- The difference between training performance and testing performance.
- The concepts of underfitting and overfitting.
- How Cross Validation helps estimate model performance more reliably.
- The role of hyperparameters in machine learning models.
- How GridSearchCV automatically searches for the best hyperparameter combination.
- How to compare baseline and tuned models.
- How to deploy an interactive Streamlit application.

---

# 🚀 How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the training script:

```bash
py train_model.py
```

Run the Streamlit application:

```bash
py -m streamlit run streamlit_app.py
```

---

# 🌐 Streamlit App

**Public App Link:**

```
https://mlb-internship-4ig5k6ykfappvtpkmjz7t6j.streamlit.app/
---

# 💻 GitHub Repository

```
https://github.com/HadeedJalani/MLB-Internship
```


---

# 👨‍💻 Developed By

**Hadeed Jalani**

MLBench Summer Internship

Day 10 – Model Evaluation & Hyperparameter Tuning
