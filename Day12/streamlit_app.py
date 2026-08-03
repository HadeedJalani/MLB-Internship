# ==========================================================
# MLBench Summer Internship - Day 12
# Fashion MNIST Classification System
# Streamlit Application
# ==========================================================

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
import matplotlib.pyplot as plt

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(

    page_title="Fashion MNIST Classification",

    page_icon="👕",

    layout="wide"

)

# --------------------------------------------------
# Load Trained Model
# --------------------------------------------------

@st.cache_resource
def load_model():

    return tf.keras.models.load_model("fashion_ann_model.keras")

model = load_model()

# --------------------------------------------------
# Class Names
# --------------------------------------------------

class_names = [

    "T-shirt / Top",

    "Trouser",

    "Pullover",

    "Dress",

    "Coat",

    "Sandal",

    "Shirt",

    "Sneaker",

    "Bag",

    "Ankle Boot"

]

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown("""

<style>

.main-title{

    text-align:center;

    font-size:45px;

    font-weight:bold;

    color:#4CAF50;

}

.sub-title{

    text-align:center;

    color:gray;

    font-size:18px;

}

.card{

    background-color:#262730;

    padding:20px;

    border-radius:12px;

    margin-bottom:20px;

}

</style>

""", unsafe_allow_html=True)

# --------------------------------------------------
# Title
# --------------------------------------------------

st.markdown(
    "<p class='main-title'>👕 Fashion MNIST Classification System</p>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='sub-title'>Deep Learning using TensorFlow / Keras</p>",
    unsafe_allow_html=True
)

st.divider()
# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.title("🧠 Project Information")

    st.markdown("---")

    st.success("### Deep Learning")

    st.write("""
Artificial Neural Network (ANN)

Dataset:
Fashion MNIST

Framework:
TensorFlow / Keras
""")

    st.info("### Dataset")

    st.write("""
Training Images : 60,000

Testing Images : 10,000

Image Size : 28 × 28

Classes : 10
""")

    st.warning("### Model")

    st.write("""
Architecture

Input Layer

↓

Flatten Layer

↓

Dense (128, ReLU)

↓

Dense (64, ReLU)

↓

Dense (10, Softmax)
""")

    st.success("### Performance")

    st.metric(

        label="Test Accuracy",

        value="87.62%"

    )

    st.metric(

        label="TensorFlow",

        value=tf.__version__

    )

    st.markdown("---")

    st.write("👨‍💻 **Developer**")

    st.write("Hadeed Jalani")

    st.caption("MLBench Summer Internship")

# --------------------------------------------------
# Dataset Overview
# --------------------------------------------------

st.header("📊 Fashion MNIST Dataset Overview")

st.write("""
Fashion MNIST is a benchmark dataset consisting of **70,000 grayscale images**
belonging to **10 clothing categories**. Each image has a size of **28 × 28 pixels**.
It is commonly used for learning and benchmarking image classification models.
""")

# Metrics
col1, col2, col3, col4 = st.columns(4)

col1.metric("Training Images", "60,000")
col2.metric("Testing Images", "10,000")
col3.metric("Classes", "10")
col4.metric("Image Size", "28 × 28")

st.divider()

# --------------------------------------------------
# Display Sample Images
# --------------------------------------------------

st.header("🖼 Sample Dataset Images")

from tensorflow.keras.datasets import fashion_mnist

(X_train, y_train), (_, _) = fashion_mnist.load_data()

fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for i, ax in enumerate(axes.flat):

    ax.imshow(X_train[i], cmap="gray")

    ax.set_title(class_names[y_train[i]], fontsize=9)

    ax.axis("off")

plt.tight_layout()

st.pyplot(fig)

st.divider()

# --------------------------------------------------
# Upload Image
# --------------------------------------------------

st.header("📤 Upload an Image")

uploaded_file = st.file_uploader(

    "Choose a clothing image",

    type=["png", "jpg", "jpeg"]

)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:

        st.subheader("Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )

    # ---------------------------------------------
    # Preprocessing
    # ---------------------------------------------

    image = image.convert("L")

    image = image.resize((28, 28))

    image_array = np.array(image)

    image_array = image_array.astype("float32") / 255.0

    image_array = image_array.reshape(1, 28, 28)

    # ---------------------------------------------
    # Prediction
    # ---------------------------------------------

    prediction = model.predict(image_array, verbose=0)

    predicted_class = np.argmax(prediction)

    confidence = float(np.max(prediction) * 100)

    with col2:

        st.subheader("Prediction")

        st.success(class_names[predicted_class])

        st.metric(

            "Confidence",

            f"{confidence:.2f}%"

        )
# --------------------------------------------------
# Prediction Probabilities
# --------------------------------------------------

    st.subheader("📊 Prediction Probabilities")

    probability_df = pd.DataFrame({

        "Category": class_names,

        "Confidence": prediction[0] * 100

    })

    fig = px.bar(

        probability_df,

        x="Confidence",

        y="Category",

        orientation="h",

        text="Confidence",

        title="Model Confidence for Each Class"

    )

    fig.update_traces(

        texttemplate="%{text:.2f}%",

        textposition="outside"

    )

    fig.update_layout(

        height=500,

        yaxis=dict(categoryorder="total ascending")

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

else:

    st.info("👆 Upload a clothing image to begin prediction.")

st.divider()

# --------------------------------------------------
# Training Performance
# --------------------------------------------------

st.header("📈 Model Training Performance")

col1, col2 = st.columns(2)

with col1:

    st.subheader("Training Accuracy")

    try:

        st.image(
            "training_accuracy.png",
            use_container_width=True
        )

    except:

        st.warning("training_accuracy.png not found.")

with col2:

    st.subheader("Training Loss")

    try:

        st.image(
            "training_loss.png",
            use_container_width=True
        )

    except:

        st.warning("training_loss.png not found.")

st.divider()

# --------------------------------------------------
# Sample Predictions
# --------------------------------------------------

st.header("🖼 Sample Predictions")

try:

    st.image(

        "sample_predictions.png",

        caption="Predicted vs Actual Labels",

        use_container_width=True

    )

except:

    st.warning("sample_predictions.png not found.")

st.divider()

# --------------------------------------------------
# About Dataset
# --------------------------------------------------

st.header("📚 About Fashion MNIST")

st.write("""

Fashion MNIST is one of the most popular datasets used to learn
Deep Learning and Image Classification.

It contains **70,000 grayscale images** of clothing items.

The dataset consists of:

- 👕 T-shirt / Top
- 👖 Trouser
- 🧥 Pullover
- 👗 Dress
- 🧥 Coat
- 👡 Sandal
- 👔 Shirt
- 👟 Sneaker
- 👜 Bag
- 🥾 Ankle Boot

Each image has a resolution of **28 × 28 pixels**.

The dataset is commonly used as a replacement for the classic handwritten digit (MNIST) dataset because it is more challenging and better represents real-world image classification tasks.

""")

st.divider()
# --------------------------------------------------
# Model Information
# --------------------------------------------------

st.header("🧠 ANN Architecture")

architecture = pd.DataFrame({

    "Layer": [

        "Input",

        "Flatten",

        "Dense",

        "Dense",

        "Output"

    ],

    "Details": [

        "28 × 28 Image",

        "784 Features",

        "128 Neurons (ReLU)",

        "64 Neurons (ReLU)",

        "10 Neurons (Softmax)"

    ]

})

st.dataframe(

    architecture,

    use_container_width=True,

    hide_index=True

)

st.divider()
# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("""

---

### 👨‍💻 Developed By

**Hadeed Jalani**

MLBench Summer Internship

Day 12 Mini Project

Artificial Neural Network using TensorFlow & Keras

""")
