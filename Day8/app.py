# ==================================================
#   STUDENT PYTHON MARKS PREDICTION SYSTEM
#        Streamlit Application
# ==================================================

import streamlit as st
import matplotlib.pyplot as plt

from utils import (
    load_dataset,
    train_model
)

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Student Python Marks Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("🎓 Student Python Marks Prediction System")

st.markdown(
    """
### MLBench Summer Internship – Day 8

This application demonstrates a complete Machine Learning workflow by predicting **Python Marks** using the following features:

- Age
- Program
- Mathematics Marks
- Statistics Marks
- Machine Learning Marks
- Attendance

### Features

- 📂 Dataset Preview
- ⚙️ Data Preprocessing
- 🤖 Linear Regression Model
- 📈 Model Evaluation
- 📊 Prediction Visualization
"""
)

# --------------------------------------------------
# Dataset Preview
# --------------------------------------------------

st.header("📂 Dataset Preview")

if st.button("Load Dataset"):

    data = load_dataset()

    st.dataframe(data)

# --------------------------------------------------
# Train Model
# --------------------------------------------------

st.header("🤖 Train Linear Regression Model")

if st.button("Train Model"):

    data, comparison, metrics = train_model()

    st.success("✅ Model Trained Successfully!")

    # -----------------------------------------
    # Metrics
    # -----------------------------------------

    st.subheader("📊 Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Mean Absolute Error",
        f"{metrics['MAE']:.2f}"
    )

    col2.metric(
        "Mean Squared Error",
        f"{metrics['MSE']:.2f}"
    )

    col3.metric(
        "R² Score",
        f"{metrics['R2']:.2f}"
    )

    # -----------------------------------------
    # Prediction Table
    # -----------------------------------------

    st.subheader("📋 Actual vs Predicted Python Marks")

    st.dataframe(comparison)

    # -----------------------------------------
    # Scatter Plot
    # -----------------------------------------

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(
        comparison["Actual Python Marks"],
        comparison["Predicted Python Marks"]
    )

    minimum = comparison["Actual Python Marks"].min()
    maximum = comparison["Actual Python Marks"].max()

    ax.plot(
        [minimum, maximum],
        [minimum, maximum],
        "r--",
        label="Perfect Prediction"
    )

    ax.set_xlabel("Actual Python Marks")
    ax.set_ylabel("Predicted Python Marks")
    ax.set_title("Actual vs Predicted Python Marks")

    ax.legend()

    st.pyplot(fig)

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("---")

st.caption(
    "MLBench Summer Internship | Day 8 | Student Python Marks Prediction System"
)