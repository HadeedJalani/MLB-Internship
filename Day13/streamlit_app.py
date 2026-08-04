# ==========================================================
# MLBench Summer Internship - Day 13
# Fashion MNIST CNN Image Classifier
# Developed by Hadeed Jalani
# ==========================================================

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Fashion MNIST CNN Classifier",
    page_icon="👕",
    layout="wide"
)

# ==========================================================
# Load CNN Model
# ==========================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cnn_model.keras")

model = load_model()

# ==========================================================
# Class Labels
# ==========================================================

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

# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.title("📘 CNN Information")

st.sidebar.success("Dataset")

st.sidebar.write("""
Fashion MNIST

• 70,000 Images

• 10 Clothing Categories

• 28 × 28 Grayscale Images
""")

st.sidebar.success("CNN Architecture")

st.sidebar.info("""
Input Image

↓

Conv2D (32)

↓

MaxPooling

↓

Conv2D (64)

↓

MaxPooling

↓

Flatten

↓

Dense (128)

↓

Dropout

↓

Softmax Output
""")

st.sidebar.success("Framework")

st.sidebar.write("TensorFlow / Keras")

# ==========================================================
# Title
# ==========================================================

st.title("👕 Fashion MNIST CNN Image Classifier")

st.markdown("""
Upload an image of clothing and let the trained
Convolutional Neural Network classify it into one of the
Fashion MNIST categories.
""")

# ==========================================================
# Upload Image
# ==========================================================

uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["png", "jpg", "jpeg"]
)

# ==========================================================
# Prediction
# ==========================================================

if uploaded_file:

    image = Image.open(uploaded_file).convert("L")
    image = image.resize((28,28))

    col1, col2 = st.columns([1,2])

    with col1:

        st.subheader("Uploaded Image")

        st.image(image, use_container_width=True)

    with col2:

        if st.button("Predict Image"):

            with st.spinner("Running CNN Prediction..."):

                img = np.array(image).astype("float32") / 255.0
                img = img.reshape(1,28,28,1)

                prediction = model.predict(img, verbose=0)

                predicted = np.argmax(prediction)

                confidence = float(np.max(prediction))

            st.success(
                f"Prediction : **{class_names[predicted]}**"
            )

            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )

            st.progress(confidence)

            st.balloons()

            # ------------------------------------------
            # Probability Chart
            # ------------------------------------------

            st.subheader("Prediction Probabilities")

            fig, ax = plt.subplots(figsize=(10,4))

            ax.bar(class_names,prediction[0])

            plt.xticks(rotation=35)

            plt.ylabel("Probability")

            plt.tight_layout()

            st.pyplot(fig)

            # ------------------------------------------
            # Probability Table
            # ------------------------------------------

            df = pd.DataFrame({

                "Class":class_names,

                "Probability (%)":
                np.round(prediction[0]*100,2)

            })

            df=df.sort_values(
                "Probability (%)",
                ascending=False
            )

            st.subheader("Detailed Prediction Scores")

            st.dataframe(
                df,
                use_container_width=True
            )

# ==========================================================
# Dataset Information
# ==========================================================

with st.expander("📊 About Fashion MNIST Dataset"):

    st.write("""

Fashion MNIST is a benchmark image classification dataset.

• 70,000 grayscale images

• 60,000 training images

• 10,000 testing images

• 10 clothing categories

• Image Size: 28 × 28 pixels

The dataset is commonly used to evaluate image
classification algorithms and deep learning models.

""")

# ==========================================================
# CNN Explanation
# ==========================================================

with st.expander("🧠 How CNN Works"):

    st.markdown("""

**Convolution Layer**

Extracts important image features.

**Pooling Layer**

Reduces image dimensions while preserving features.

**Flatten Layer**

Converts feature maps into vectors.

**Dense Layer**

Learns high-level patterns.

**Softmax Layer**

Outputs probabilities for each clothing category.

""")

# ==========================================================
# Fashion MNIST Categories
# ==========================================================

st.subheader("Fashion MNIST Classes")

table = pd.DataFrame({

    "Index":list(range(10)),

    "Category":class_names

})

st.table(table)

# ==========================================================
# Footer
# ==========================================================

st.markdown("---")

st.caption("""
MLBench Summer Internship • Day 13

Fashion MNIST CNN Image Classifier

Developed by **Hadeed Jalani**
""")