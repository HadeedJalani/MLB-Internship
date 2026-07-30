import streamlit as st
import numpy as np
import joblib

# ==================================================
# Page Configuration
# ==================================================

st.set_page_config(
    page_title="Iris Flower Species Predictor",
    page_icon="🌸",
    layout="wide",
)

# ==================================================
# Load Model
# ==================================================

model = joblib.load("logistic_regression_model.pkl")

species = [
    "Setosa",
    "Versicolor",
    "Virginica",
]

flower_icons = {
    "Setosa": "🌸",
    "Versicolor": "🌼",
    "Virginica": "🌺",
}

# ==================================================
# Title
# ==================================================

st.title("🌸 Iris Flower Species Predictor")

st.write(
    """
This application predicts the species of an Iris flower
using a trained Logistic Regression machine learning model.

Adjust the flower measurements below to see the prediction update instantly.
"""
)

# ==================================================
# Sidebar
# ==================================================

st.sidebar.title("📌 Project Information")

st.sidebar.markdown("---")

st.sidebar.subheader("Algorithm")
st.sidebar.success("Logistic Regression")

st.sidebar.subheader("Dataset")
st.sidebar.info("Scikit-Learn Iris Dataset")

st.sidebar.subheader("Dataset Statistics")

st.sidebar.write("Training Samples : 120")
st.sidebar.write("Testing Samples : 30")
st.sidebar.write("Total Samples : 150")
st.sidebar.write("Features : 4")
st.sidebar.write("Classes : 3")

st.sidebar.subheader("Model Accuracy")

st.sidebar.success("96.67 %")

st.sidebar.markdown("---")

st.sidebar.write("Developed By")

st.sidebar.write("**Hadeed Jalani**")

# ==================================================
# Layout
# ==================================================

left, right = st.columns([2, 1])

# ==================================================
# Input Sliders
# ==================================================

with left:

    st.subheader("Flower Measurements")

    sepal_length = st.slider(
        "Sepal Length (cm)",
        4.0,
        8.0,
        5.8,
        0.1,
    )

    sepal_width = st.slider(
        "Sepal Width (cm)",
        2.0,
        4.5,
        3.0,
        0.1,
    )

    petal_length = st.slider(
        "Petal Length (cm)",
        1.0,
        7.0,
        4.0,
        0.1,
    )

    petal_width = st.slider(
        "Petal Width (cm)",
        0.1,
        2.5,
        1.2,
        0.1,
    )

# ==================================================
# Prediction
# ==================================================

features = np.array([
    [
        sepal_length,
        sepal_width,
        petal_length,
        petal_width,
    ]
])

prediction = model.predict(features)

probability = model.predict_proba(features)

predicted_species = species[prediction[0]]

confidence = np.max(probability) * 100

# ==================================================
# Prediction Panel
# ==================================================

with right:

    st.subheader("Prediction")

    st.success(
        f"{flower_icons[predicted_species]} {predicted_species}"
    )

    st.metric(
        label="Confidence",
        value=f"{confidence:.2f}%"
    )

    st.subheader("Prediction Probability")

    st.write(f"🌸 Setosa : {probability[0][0]*100:.2f}%")
    st.progress(float(probability[0][0]))

    st.write(f"🌼 Versicolor : {probability[0][1]*100:.2f}%")
    st.progress(float(probability[0][1]))

    st.write(f"🌺 Virginica : {probability[0][2]*100:.2f}%")
    st.progress(float(probability[0][2]))

# ==================================================
# Feature Summary
# ==================================================

st.markdown("---")

st.subheader("Selected Feature Values")

st.table(
    {
        "Feature": [
            "Sepal Length (cm)",
            "Sepal Width (cm)",
            "Petal Length (cm)",
            "Petal Width (cm)",
        ],
        "Value": [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width,
        ],
    }
)

# ==================================================
# Footer
# ==================================================

st.markdown("---")

st.info(
    "This prediction is generated using a Logistic Regression model trained on the Scikit-Learn Iris Dataset."
)