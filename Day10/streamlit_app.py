# ==================================================
#      BREAST CANCER PREDICTION SYSTEM
#      MLBench Summer Internship - Day 10
# ==================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Breast Cancer Prediction System",
    page_icon="🩺",
    layout="wide",
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(
    os.path.join(BASE_DIR, "logistic_regression_model.pkl")
)

scaler = joblib.load(
    os.path.join(BASE_DIR, "scaler.pkl")
)
# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

data = load_breast_cancer()

df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

df["target"] = data.target

feature_names = data.feature_names

target_names = data.target_names

# --------------------------------------------------
# PAGE TITLE
# --------------------------------------------------

st.title("🩺 Breast Cancer Prediction System")

st.write(
    """
This application predicts whether a tumor is **Benign** or **Malignant**
using a Logistic Regression model trained on the
Breast Cancer Wisconsin Diagnostic Dataset.
"""
)

st.markdown("---")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("📌 Project Information")

st.sidebar.markdown("---")

st.sidebar.subheader("Model")

st.sidebar.success("Logistic Regression")

st.sidebar.subheader("Dataset")

st.sidebar.info("Breast Cancer Wisconsin Diagnostic Dataset")

st.sidebar.subheader("Training Samples")

st.sidebar.write("455")

st.sidebar.subheader("Testing Samples")

st.sidebar.write("114")

st.sidebar.subheader("Model Accuracy")

st.sidebar.success("97%")

st.sidebar.markdown("---")

st.sidebar.write("Developed By")

st.sidebar.write("**Hadeed Jalani**")

st.sidebar.markdown("---")

st.sidebar.write(
    """
This application demonstrates

✔ Dataset Exploration

✔ Logistic Regression

✔ Model Evaluation

✔ Hyperparameter Tuning

✔ Live Prediction
"""
)

# --------------------------------------------------
# DATASET UPLOAD
# --------------------------------------------------

st.header("📂 Dataset Upload")

uploaded_file = st.file_uploader(
    "Upload a CSV Dataset (Optional)",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        uploaded_df = pd.read_csv(uploaded_file)

        st.success("Dataset uploaded successfully.")

        current_df = uploaded_df

    except Exception:

        st.error("Unable to read CSV file.")

        current_df = df

else:

    st.info(
        "No dataset uploaded. Using the built-in Breast Cancer Wisconsin Dataset."
    )

    current_df = df

st.markdown("---")

# --------------------------------------------------
# DATASET PREVIEW
# --------------------------------------------------

st.header("📋 Dataset Preview")

st.write("Shape :", current_df.shape)

st.dataframe(
    current_df.head(10)
)

st.markdown("---")

# --------------------------------------------------
# DATASET INFORMATION
# --------------------------------------------------

st.header("📊 Dataset Information")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Rows",
        current_df.shape[0]
    )

    st.metric(
        "Columns",
        current_df.shape[1]
    )

with col2:

    if "target" in current_df.columns:

        st.metric(
            "Malignant",
            int((current_df["target"] == 0).sum())
        )

        st.metric(
            "Benign",
            int((current_df["target"] == 1).sum())
        )

st.markdown("---")
# --------------------------------------------------
# PREDICTION SECTION
# --------------------------------------------------

st.header("🩺 Breast Cancer Prediction")

prediction_mode = st.radio(
    "Choose Prediction Method",
    (
        "🎚 Slider Mode (Live)",
        "⌨ Manual Input",
    )
)

st.markdown("---")

# ==================================================
# SLIDER MODE
# ==================================================

if prediction_mode == "🎚 Slider Mode (Live)":

    st.subheader("Move the sliders to predict instantly")

    values = []

    columns = st.columns(2)

    for index, feature in enumerate(feature_names):

        minimum = float(df[feature].min())
        maximum = float(df[feature].max())
        average = float(df[feature].mean())

        with columns[index % 2]:

            value = st.slider(
                feature.title(),
                min_value=minimum,
                max_value=maximum,
                value=average,
                step=(maximum - minimum) / 100,
            )

        values.append(value)

    sample = np.array(values).reshape(1, -1)

    sample_scaled = scaler.transform(sample)

    prediction = model.predict(sample_scaled)[0]

    probability = model.predict_proba(sample_scaled)[0]

# ==================================================
# MANUAL INPUT MODE
# ==================================================

else:

    st.subheader("Enter feature values manually")

    values = []

    columns = st.columns(2)

    for index, feature in enumerate(feature_names):

        average = float(df[feature].mean())

        with columns[index % 2]:

            value = st.number_input(
                feature.title(),
                value=average,
                format="%.4f",
            )

        values.append(value)

    sample = np.array(values).reshape(1, -1)

    sample_scaled = scaler.transform(sample)

    if st.button("Predict"):

        prediction = model.predict(sample_scaled)[0]

        probability = model.predict_proba(sample_scaled)[0]

    else:

        prediction = None

# --------------------------------------------------
# DISPLAY PREDICTION
# --------------------------------------------------

if prediction is not None:

    st.markdown("---")

    st.header("📊 Prediction Result")

    left, right = st.columns([1, 1])

    # ----------------------------------------------
    # LEFT COLUMN
    # ----------------------------------------------

    with left:

        if prediction == 1:

            st.success("🟢 BENIGN TUMOR")

            st.write(
                """
The trained Logistic Regression model predicts
that the tumor is **Benign**.
"""
            )

        else:

            st.error("🔴 MALIGNANT TUMOR")

            st.write(
                """
The trained Logistic Regression model predicts
that the tumor is **Malignant**.
"""
            )

        confidence = max(probability) * 100

        st.metric(
            label="Prediction Confidence",
            value=f"{confidence:.2f}%"
        )

    # ----------------------------------------------
    # RIGHT COLUMN
    # ----------------------------------------------

    with right:

        st.subheader("Prediction Probability")

        st.write("Malignant")

        st.progress(float(probability[0]))

        st.write(
            f"{probability[0]*100:.2f}%"
        )

        st.write("Benign")

        st.progress(float(probability[1]))

        st.write(
            f"{probability[1]*100:.2f}%"
        )

    st.markdown("---")

# --------------------------------------------------
# FEATURE SUMMARY
# --------------------------------------------------

if prediction is not None:

    st.header("📋 Selected Feature Values")

    feature_table = pd.DataFrame({

        "Feature": feature_names,

        "Input Value": values,

    })

    st.dataframe(
        feature_table,
        height=500
    )

    st.markdown("---")
# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

st.header("📈 Model Evaluation")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

scaler_app = StandardScaler()

X_train_scaled = scaler_app.fit_transform(X_train)

X_test_scaled = scaler_app.transform(X_test)

# --------------------------------------------------
# BASELINE MODEL
# --------------------------------------------------

baseline_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
)

baseline_model.fit(
    X_train_scaled,
    y_train,
)

baseline_predictions = baseline_model.predict(
    X_test_scaled
)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions,
)

baseline_precision = precision_score(
    y_test,
    baseline_predictions,
)

baseline_recall = recall_score(
    y_test,
    baseline_predictions,
)

baseline_f1 = f1_score(
    y_test,
    baseline_predictions,
)

baseline_confusion = confusion_matrix(
    y_test,
    baseline_predictions,
)

st.subheader("Baseline Logistic Regression")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Accuracy",
    f"{baseline_accuracy:.4f}"
)

c2.metric(
    "Precision",
    f"{baseline_precision:.4f}"
)

c3.metric(
    "Recall",
    f"{baseline_recall:.4f}"
)

c4.metric(
    "F1 Score",
    f"{baseline_f1:.4f}"
)

st.markdown("---")

# --------------------------------------------------
# GRID SEARCH
# --------------------------------------------------

from sklearn.model_selection import GridSearchCV

st.subheader("🔍 Hyperparameter Tuning")

parameter_grid = {

    "C": [
        0.01,
        0.1,
        1,
        10,
        100,
    ],

    "solver": [
        "liblinear",
        "lbfgs",
    ],
}

grid = GridSearchCV(

    LogisticRegression(
        max_iter=1000,
        random_state=42,
    ),

    parameter_grid,

    cv=5,

    scoring="accuracy",

)

grid.fit(
    X_train_scaled,
    y_train,
)

best_model = grid.best_estimator_

tuned_predictions = best_model.predict(
    X_test_scaled
)

tuned_accuracy = accuracy_score(
    y_test,
    tuned_predictions,
)

tuned_precision = precision_score(
    y_test,
    tuned_predictions,
)

tuned_recall = recall_score(
    y_test,
    tuned_predictions,
)

tuned_f1 = f1_score(
    y_test,
    tuned_predictions,
)

tuned_confusion = confusion_matrix(
    y_test,
    tuned_predictions,
)

st.success("Best Parameters Found")

st.write(grid.best_params_)

st.write(
    f"Cross Validation Score : {grid.best_score_:.4f}"
)

st.markdown("---")

# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------

st.subheader("🏆 Baseline vs Tuned Comparison")

comparison = pd.DataFrame({

    "Metric":[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
    ],

    "Baseline":[
        baseline_accuracy,
        baseline_precision,
        baseline_recall,
        baseline_f1,
    ],

    "Tuned":[
        tuned_accuracy,
        tuned_precision,
        tuned_recall,
        tuned_f1,
    ],

})

st.table(
    comparison.round(4)
)

st.markdown("---")
# --------------------------------------------------
# CONFUSION MATRICES
# --------------------------------------------------

st.header("📊 Confusion Matrices")

left, right = st.columns(2)

with left:

    st.subheader("Baseline Model")

    baseline_cm = pd.DataFrame(

        baseline_confusion,

        index=["Actual Malignant", "Actual Benign"],

        columns=["Predicted Malignant", "Predicted Benign"]

    )

    st.dataframe(baseline_cm)

with right:

    st.subheader("Tuned Model")

    tuned_cm = pd.DataFrame(

        tuned_confusion,

        index=["Actual Malignant", "Actual Benign"],

        columns=["Predicted Malignant", "Predicted Benign"]

    )

    st.dataframe(tuned_cm)

st.markdown("---")

# --------------------------------------------------
# CLASSIFICATION REPORTS
# --------------------------------------------------

st.header("📄 Classification Reports")

left, right = st.columns(2)

with left:

    st.subheader("Baseline Model")

    baseline_report = classification_report(

        y_test,

        baseline_predictions,

    )

    st.text(baseline_report)

with right:

    st.subheader("Tuned Model")

    tuned_report = classification_report(

        y_test,

        tuned_predictions,

    )

    st.text(tuned_report)

st.markdown("---")

# --------------------------------------------------
# DOWNLOAD RESULTS
# --------------------------------------------------

st.header("📥 Download Results")

comparison_download = comparison.round(4)

csv = comparison_download.to_csv(index=False)

st.download_button(

    label="⬇ Download Comparison Report",

    data=csv,

    file_name="model_comparison.csv",

    mime="text/csv",

)
# --------------------------------------------------
# PROJECT SUMMARY
# --------------------------------------------------

st.markdown("---")

st.header("📌 Project Summary")

best_model_name = (

    "Tuned Logistic Regression"

    if tuned_accuracy >= baseline_accuracy

    else "Baseline Logistic Regression"

)

st.success(f"🏆 Best Model : {best_model_name}")

st.write("### Observations")

st.write(
"""
• Logistic Regression performs exceptionally well on the Breast Cancer dataset.

• GridSearchCV searches different combinations of hyperparameters and selects the best performing model.

• Cross Validation helps reduce overfitting and provides a more reliable estimate of model performance.

• Hyperparameter tuning slightly improves the model's generalization performance.

• The tuned model is recommended for future predictions.
"""
)

st.markdown("---")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.markdown(
"""
### 👨‍💻 Developed By

**Hadeed Jalani**

MLBench Summer Internship — Day 10

Breast Cancer Prediction System using Logistic Regression & GridSearchCV
"""
)
