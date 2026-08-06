# ==========================================================
# MLBench Summer Internship - Day 14
# Cats vs Dogs Image Classifier
# Transfer Learning using MobileNetV2
# Professional AI Dashboard
# ==========================================================

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from PIL import Image, ImageEnhance, ImageOps
from pathlib import Path
import random
import os
from datetime import datetime

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Cats vs Dogs AI",
    page_icon="🐶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main{
    background:#f5f7fa;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

.hero{
    background:linear-gradient(90deg,#0f172a,#1e40af);
    padding:30px;
    border-radius:20px;
    color:white;
}

.metric-card{
    background:white;
    padding:18px;
    border-radius:15px;
    box-shadow:0 4px 12px rgba(0,0,0,.12);
}

.sidebar .sidebar-content{
    background:#111827;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# MODEL
# ==========================================================

MODEL_PATH = "cats_vs_dogs_model.keras"

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(MODEL_PATH)

    return model

model = load_model()

# ==========================================================
# CLASS NAMES
# ==========================================================

class_names = [

    "Cat",

    "Dog"

]

# ==========================================================
# SESSION STATE
# ==========================================================

if "history" not in st.session_state:

    st.session_state.history=[]

if "cats" not in st.session_state:

    st.session_state.cats=0

if "dogs" not in st.session_state:

    st.session_state.dogs=0

if "total" not in st.session_state:

    st.session_state.total=0

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/616/616408.png",
    width=120
)

st.sidebar.title("AI Dashboard")

page=st.sidebar.radio(

    "Navigation",

    [

        "🏠 Dashboard",

        "🤖 Prediction",

        "📊 Analytics",

        "📂 Dataset Explorer",

        "🧠 Model Details",

        "ℹ About"

    ]

)

st.sidebar.markdown("---")

st.sidebar.success("Transfer Learning")

st.sidebar.info("MobileNetV2")

st.sidebar.metric(

    "Model Accuracy",

    "99.22%"

)

# ==========================================================
# DASHBOARD
# ==========================================================

if page=="🏠 Dashboard":

    st.markdown("""

<div class="hero">

<h1>🐶 Cats vs Dogs Image Classifier</h1>

<h3>Transfer Learning using MobileNetV2</h3>

<p>

Professional Deep Learning Dashboard

</p>

</div>

""",unsafe_allow_html=True)

    st.write("")

    c1,c2,c3,c4=st.columns(4)

    with c1:

        st.metric(

            "Model",

            "MobileNetV2"

        )

    with c2:

        st.metric(

            "Accuracy",

            "99.22%"

        )

    with c3:

        st.metric(

            "Classes",

            "2"

        )

    with c4:

        st.metric(

            "Dataset",

            "25,000 Images"

        )

    st.write("")

    left,right=st.columns([2,1])

    with left:

        st.subheader("Project Overview")

        st.write("""

This project demonstrates **Transfer Learning**
using **MobileNetV2** for binary image classification.

The model has been trained to distinguish between

- 🐱 Cats

- 🐶 Dogs

using a dataset containing over **25,000 images**.

Instead of training a CNN from scratch,
MobileNetV2 provides pre-trained image features,
allowing faster convergence and higher accuracy.

""")

    with right:

        fig=go.Figure(go.Indicator(

            mode="gauge+number",

            value=99.22,

            title={"text":"Accuracy"},

            gauge={

                "axis":{"range":[0,100]},

                "bar":{"thickness":0.4},

                "steps":[

                    {"range":[0,60],"color":"#ef4444"},

                    {"range":[60,80],"color":"orange"},

                    {"range":[80,100],"color":"green"}

                ]

            }

        ))

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    st.write("")

    st.subheader("Transfer Learning Pipeline")

    st.info("""

Image

↓

Data Augmentation

↓

MobileNetV2

↓

Global Average Pooling

↓

Dense(256)

↓

Dropout

↓

Output Layer

↓

Prediction

""")

# ==========================================================
# PREDICTION PAGE
# ==========================================================

elif page == "🤖 Prediction":

    st.title("🤖 AI Image Prediction Studio")

    st.write(
        """
Upload a **Cat** or **Dog** image and use the image editing tools
before sending it to the AI model.

The processed image will be used for prediction.
"""
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "📤 Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        original_image = Image.open(uploaded_file).convert("RGB")

        # --------------------------------------------------
        # IMAGE CONTROLS
        # --------------------------------------------------

        st.subheader("🎛 Image Controls")

        col1, col2 = st.columns(2)

        with col1:

            brightness = st.slider(
                "Brightness",
                0.5,
                2.0,
                1.0,
                0.1
            )

            contrast = st.slider(
                "Contrast",
                0.5,
                2.0,
                1.0,
                0.1
            )

            sharpness = st.slider(
                "Sharpness",
                0.5,
                3.0,
                1.0,
                0.1
            )

        with col2:

            rotation = st.slider(
                "Rotation",
                -180,
                180,
                0
            )

            grayscale = st.checkbox(
                "Convert to Grayscale"
            )

            flip = st.selectbox(

                "Flip",

                [

                    "None",

                    "Horizontal",

                    "Vertical"

                ]

            )

        resize = st.slider(

            "Resize",

            150,

            500,

            224

        )

        # --------------------------------------------------
        # APPLY CHANGES
        # --------------------------------------------------

        processed = original_image.copy()

        processed = ImageEnhance.Brightness(
            processed
        ).enhance(brightness)

        processed = ImageEnhance.Contrast(
            processed
        ).enhance(contrast)

        processed = ImageEnhance.Sharpness(
            processed
        ).enhance(sharpness)

        processed = processed.rotate(rotation)

        if flip == "Horizontal":

            processed = ImageOps.mirror(processed)

        elif flip == "Vertical":

            processed = ImageOps.flip(processed)

        if grayscale:

            processed = ImageOps.grayscale(processed).convert("RGB")

        processed = processed.resize(
            (resize, resize)
        )

        # --------------------------------------------------
        # DISPLAY
        # --------------------------------------------------

        st.divider()

        left, right = st.columns(2)

        with left:

            st.subheader("📷 Original")

            st.image(
                original_image,
                use_container_width=True
            )

        with right:

            st.subheader("✨ Processed")

            st.image(
                processed,
                use_container_width=True
            )

        st.divider()

        st.subheader("📝 Image Information")

        info1, info2, info3 = st.columns(3)

        with info1:

            st.metric(

                "Width",

                processed.width

            )

        with info2:

            st.metric(

                "Height",

                processed.height

            )

        with info3:

            st.metric(

                "Mode",

                processed.mode

            )

        # --------------------------------------------------
        # PREDICT BUTTON
        # --------------------------------------------------

        if st.button(
            "🚀 Predict Image",
            use_container_width=True
        ):

            img = processed.resize((224,224))

            img = np.array(img)

            img = img.astype(np.float32)

            img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

            img = np.expand_dims(img, axis=0)

            with st.spinner("Running AI Prediction..."):

                prediction = model.predict(
                    img,
                    verbose=0
                )

            #probabilities = tf.nn.softmax(prediction[0]).numpy()

            predicted_class = np.argmax(probabilities)

            confidence = probabilities[predicted_class] * 100

            st.session_state.total += 1

            if predicted_class == 0:

                st.session_state.cats += 1

            else:

                st.session_state.dogs += 1

            st.session_state.history.append({

                "Prediction": class_names[predicted_class],

                "Confidence": round(confidence,2),

                "Time": datetime.now().strftime("%H:%M:%S")

            })

            st.success(

                f"### 🏆 Prediction: {class_names[predicted_class]}"

            )

            st.metric(

                "Confidence",

                f"{confidence:.2f}%"

            )

            st.progress(float(confidence/100))

            # ======================================================
            # PROBABILITY RESULTS
            # ======================================================

            st.divider()

            st.subheader("📊 Prediction Probabilities")

            probability_df = pd.DataFrame({

                "Class": class_names,

                "Probability": probabilities * 100

            })

            fig = px.bar(

                probability_df,

                x="Class",

                y="Probability",

                color="Probability",

                text="Probability",

                color_continuous_scale="Blues"

            )

            fig.update_traces(

                texttemplate="%{text:.2f}%",

                textposition="outside"

            )

            fig.update_layout(

                height=450,

                xaxis_title="Class",

                yaxis_title="Confidence (%)",

                showlegend=False

            )

            st.plotly_chart(

                fig,

                use_container_width=True

            )

            # ======================================================
            # PIE CHART
            # ======================================================

            st.subheader("🥧 Confidence Distribution")

            pie = px.pie(

                values=probabilities,

                names=class_names,

                hole=0.45

            )

            pie.update_layout(

                height=450

            )

            st.plotly_chart(

                pie,

                use_container_width=True

            )

            # ======================================================
            # CONFIDENCE LEVEL
            # ======================================================

            st.subheader("🎯 Prediction Confidence")

            if confidence >= 98:

                st.success(

                    "🟢 Excellent Confidence"

                )

            elif confidence >= 90:

                st.info(

                    "🔵 High Confidence"

                )

            elif confidence >= 75:

                st.warning(

                    "🟡 Moderate Confidence"

                )

            else:

                st.error(

                    "🔴 Low Confidence"

                )

            # ======================================================
            # AI EXPLANATION
            # ======================================================

            st.subheader("🧠 AI Interpretation")

            if predicted_class == 0:

                st.write("""

The model predicts this image as a **CAT**.

This decision is based on learned visual
features such as:

- Face Shape

- Ear Structure

- Fur Texture

- Body Proportions

- Eye Position

Transfer Learning enables MobileNetV2 to
recognize these patterns very effectively.

""")

            else:

                st.write("""

The model predicts this image as a **DOG**.

The prediction is influenced by:

- Snout Shape

- Ear Position

- Body Size

- Fur Pattern

- Facial Features

These visual characteristics were learned
through Transfer Learning using MobileNetV2.

""")

            # ======================================================
            # CONFIDENCE THRESHOLD
            # ======================================================

            st.subheader("⚙ Confidence Threshold")

            threshold = st.slider(

                "Minimum Confidence",

                50,

                100,

                80

            )

            if confidence >= threshold:

                st.success(

                    "Prediction exceeds selected confidence threshold."

                )

            else:

                st.warning(

                    "Prediction is below selected confidence threshold."

                )

            # ======================================================
            # DOWNLOAD REPORT
            # ======================================================

            report = f"""

Cats vs Dogs AI Report

==========================

Prediction : {class_names[predicted_class]}

Confidence : {confidence:.2f} %

Date : {datetime.now().strftime("%d-%m-%Y")}

Time : {datetime.now().strftime("%H:%M:%S")}

Model : MobileNetV2

Transfer Learning : Enabled

Image Size : {processed.width} x {processed.height}

"""

            st.download_button(

                "📥 Download Prediction Report",

                report,

                file_name="prediction_report.txt"

            )

# ==========================================================
# ANALYTICS PAGE
# ==========================================================

elif page == "📊 Analytics":

    st.title("📊 AI Analytics Dashboard")

    st.write(
        "Track model usage and prediction statistics from the current session."
    )

    st.divider()

    # ======================================================
    # METRICS
    # ======================================================

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Total Predictions",
            st.session_state.total
        )

    with c2:
        st.metric(
            "Cats Predicted",
            st.session_state.cats
        )

    with c3:
        st.metric(
            "Dogs Predicted",
            st.session_state.dogs
        )

    with c4:

        if st.session_state.total == 0:

            avg = 0

        else:

            avg = np.mean(

                [x["Confidence"] for x in st.session_state.history]

            )

        st.metric(

            "Average Confidence",

            f"{avg:.2f}%"

        )

    st.divider()

    # ======================================================
    # PIE CHART
    # ======================================================

    st.subheader("Prediction Distribution")

    pie = px.pie(

        values=[

            st.session_state.cats,

            st.session_state.dogs

        ],

        names=[

            "Cats",

            "Dogs"

        ],

        hole=0.45

    )

    st.plotly_chart(

        pie,

        use_container_width=True

    )

    # ======================================================
    # BAR CHART
    # ======================================================

    st.subheader("Prediction Count")

    chart = pd.DataFrame({

        "Class":[

            "Cats",

            "Dogs"

        ],

        "Predictions":[

            st.session_state.cats,

            st.session_state.dogs

        ]

    })

    fig = px.bar(

        chart,

        x="Class",

        y="Predictions",

        color="Predictions",

        text="Predictions"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    # ======================================================
    # HISTORY TABLE
    # ======================================================

    st.subheader("Prediction History")

    if len(st.session_state.history) > 0:

        history_df = pd.DataFrame(

            st.session_state.history

        )

        st.dataframe(

            history_df,

            use_container_width=True

        )

    else:

        st.info(

            "No predictions made yet."

        )

    # ======================================================
    # CONFIDENCE TREND
    # ======================================================

    if len(st.session_state.history) > 0:

        st.subheader("Confidence Trend")

        trend = history_df.copy()

        trend["Prediction #"] = np.arange(

            1,

            len(trend)+1

        )

        fig = px.line(

            trend,

            x="Prediction #",

            y="Confidence",

            markers=True

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    # ======================================================
    # DOWNLOAD HISTORY
    # ======================================================

    if len(st.session_state.history) > 0:

        csv = history_df.to_csv(index=False)

        st.download_button(

            "📥 Download Prediction History",

            csv,

            "prediction_history.csv",

            "text/csv"

        )

    # ======================================================
    # CLEAR HISTORY
    # ======================================================

    if st.button("🗑 Clear Session History"):

        st.session_state.history = []

        st.session_state.cats = 0

        st.session_state.dogs = 0

        st.session_state.total = 0

        st.success(

            "Session history cleared."

        )

        st.rerun()

# ==========================================================
# DATASET EXPLORER
# ==========================================================

elif page == "📂 Dataset Explorer":

    st.title("📂 Cats vs Dogs Dataset Explorer")

    st.write(
        """
Browse the training dataset used for Transfer Learning.
You can inspect random samples and view dataset statistics.
"""
    )

    st.divider()

    TRAIN_DIR = Path("dataset/train")

    CAT_DIR = TRAIN_DIR / "cats"
    DOG_DIR = TRAIN_DIR / "dogs"

    if not TRAIN_DIR.exists():

        st.error("Dataset folder not found.")

    else:

        cat_images = list(CAT_DIR.glob("*"))
        dog_images = list(DOG_DIR.glob("*"))

        # ======================================================
        # DATASET METRICS
        # ======================================================

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(

                "Cat Images",

                len(cat_images)

            )

        with c2:

            st.metric(

                "Dog Images",

                len(dog_images)

            )

        with c3:

            st.metric(

                "Total Images",

                len(cat_images) + len(dog_images)

            )

        st.divider()

        # ======================================================
        # RANDOM IMAGE BUTTONS
        # ======================================================

        col1, col2 = st.columns(2)

        with col1:

            if st.button("🐱 Show Random Cat"):

                random_cat = random.choice(cat_images)

                st.image(

                    str(random_cat),

                    caption=random_cat.name,

                    use_container_width=True

                )

        with col2:

            if st.button("🐶 Show Random Dog"):

                random_dog = random.choice(dog_images)

                st.image(

                    str(random_dog),

                    caption=random_dog.name,

                    use_container_width=True

                )

        st.divider()

        # ======================================================
        # RANDOM GALLERY
        # ======================================================

        st.subheader("🖼 Random Dataset Gallery")

        gallery = st.slider(

            "Number of Images",

            4,

            12,

            8

        )

        images = random.sample(

            cat_images + dog_images,

            min(gallery, len(cat_images + dog_images))

        )

        cols = st.columns(4)

        for i, img in enumerate(images):

            with cols[i % 4]:

                st.image(

                    str(img),

                    use_container_width=True

                )

                if "cat" in img.name.lower():

                    st.caption("🐱 Cat")

                else:

                    st.caption("🐶 Dog")

        st.divider()

        # ======================================================
        # DATASET DISTRIBUTION
        # ======================================================

        st.subheader("📊 Dataset Distribution")

        distribution = pd.DataFrame({

            "Class":[

                "Cats",

                "Dogs"

            ],

            "Images":[

                len(cat_images),

                len(dog_images)

            ]

        })

        fig = px.bar(

            distribution,

            x="Class",

            y="Images",

            color="Images",

            text="Images"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

        st.divider()

        # ======================================================
        # DATASET INFO
        # ======================================================

        st.subheader("📑 Dataset Information")

        st.info(f"""

Dataset Root

{TRAIN_DIR}

Classes

• Cats

• Dogs

Image Size Used

224 × 224

Transfer Learning Model

MobileNetV2

""")

        # ======================================================
        # COMPARE TWO RANDOM IMAGES
        # ======================================================

        st.subheader("🔍 Compare Samples")

        if st.button("Generate Comparison"):

            cat = random.choice(cat_images)

            dog = random.choice(dog_images)

            left, right = st.columns(2)

            with left:

                st.image(

                    str(cat),

                    caption="🐱 Cat",

                    use_container_width=True

                )

            with right:

                st.image(

                    str(dog),

                    caption="🐶 Dog",

                    use_container_width=True

                )

# ==========================================================
# MODEL DETAILS
# ==========================================================

elif page == "🧠 Model Details":

    st.title("🧠 Model Details & AI Explainability")

    st.write(
        """
Explore the architecture, training results, and performance of the
Transfer Learning model.
"""
    )

    st.divider()

    # ======================================================
    # MODEL INFORMATION
    # ======================================================

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Architecture", "MobileNetV2")

    with c2:
        st.metric("Input Size", "224 × 224")

    with c3:
        st.metric("Classes", "2")

    with c4:
        st.metric("Validation Accuracy", "99.22%")

    st.divider()

    # ======================================================
    # MODEL DESCRIPTION
    # ======================================================

    st.subheader("📖 Model Overview")

    st.write("""

### MobileNetV2

MobileNetV2 is a lightweight Convolutional Neural Network designed
for high accuracy with low computational cost.

Instead of training millions of parameters from scratch,
Transfer Learning allows us to reuse features learned from ImageNet.

Our model workflow:

• Input Image (224×224)

↓

• Data Augmentation

↓

• MobileNetV2 Backbone

↓

• Global Average Pooling

↓

• Dense (256)

↓

• Dropout

↓

• Softmax Output Layer

""")

    st.divider()

    # ======================================================
    # PARAMETERS
    # ======================================================

    st.subheader("⚙ Model Statistics")

    trainable = sum(
        np.prod(v.shape)
        for v in model.trainable_variables
    )

    non_trainable = sum(
        np.prod(v.shape)
        for v in model.non_trainable_variables
    )

    total = trainable + non_trainable

    parameter_df = pd.DataFrame({

        "Category":[

            "Trainable",

            "Frozen",

            "Total"

        ],

        "Parameters":[

            trainable,

            non_trainable,

            total

        ]

    })

    fig = px.bar(

        parameter_df,

        x="Category",

        y="Parameters",

        color="Category",

        text="Parameters"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.divider()

    # ======================================================
    # TRAINING RESULTS
    # ======================================================

    st.subheader("📈 Training Results")

    left, right = st.columns(2)

    with left:

        if os.path.exists("accuracy_curve.png"):

            st.image(

                "accuracy_curve.png",

                caption="Training Accuracy",

                use_container_width=True

            )

        else:

            st.warning("accuracy_curve.png not found.")

    with right:

        if os.path.exists("loss_curve.png"):

            st.image(

                "loss_curve.png",

                caption="Training Loss",

                use_container_width=True

            )

        else:

            st.warning("loss_curve.png not found.")

    st.divider()

    # ======================================================
    # CONFUSION MATRIX
    # ======================================================

    st.subheader("📊 Confusion Matrix")

    if os.path.exists("confusion_matrix.png"):

        st.image(

            "confusion_matrix.png",

            use_container_width=True

        )

    else:

        st.warning("confusion_matrix.png not found.")

    st.divider()

    # ======================================================
    # PERFORMANCE
    # ======================================================

    st.subheader("🏆 Model Performance")

    performance = pd.DataFrame({

        "Metric": [

            "Training Accuracy",

            "Validation Accuracy",

            "Test Accuracy"

        ],

        "Value": [

            99.60,

            99.39,

            99.22

        ]

    })

    chart = px.bar(

        performance,

        x="Metric",

        y="Value",

        color="Value",

        text="Value"

    )

    chart.update_traces(

        texttemplate="%{text:.2f}%",

        textposition="outside"

    )

    st.plotly_chart(

        chart,

        use_container_width=True

    )

    st.divider()

    # ======================================================
    # MODEL SUMMARY
    # ======================================================

    st.subheader("📄 Model Summary")

    summary = []

    model.summary(

        print_fn=lambda x: summary.append(x)

    )

    st.code(

        "\n".join(summary),

        language="text"

    )

    st.divider()

    # ======================================================
    # DOWNLOAD MODEL
    # ======================================================

    st.subheader("💾 Download Model")

    if os.path.exists("cats_vs_dogs_model.keras"):

        with open("cats_vs_dogs_model.keras", "rb") as f:

            st.download_button(

                "⬇ Download Trained Model",

                data=f,

                file_name="cats_vs_dogs_model.keras"

            )

    else:

        st.warning("Model file not found.")

    st.divider()

    # ======================================================
    # ACHIEVEMENTS
    # ======================================================

    st.success("""

### 🎉 Training Achievements

✅ Transfer Learning Implemented

✅ MobileNetV2 Backbone

✅ Data Augmentation

✅ Feature Extraction

✅ Fine-Tuning

✅ Validation Accuracy > 99%

✅ Professional Streamlit Dashboard

""")

# ==========================================================
# ABOUT PAGE
# ==========================================================

elif page == "ℹ About":

    st.title("ℹ About This Project")

    st.markdown("""
This application demonstrates **Transfer Learning** using **MobileNetV2**
for binary image classification (Cats vs Dogs).

The project was developed as part of the **MLBench Summer Internship – Day 14**.
""")

    st.divider()

    # ======================================================
    # PROJECT OVERVIEW
    # ======================================================

    st.header("📌 Project Overview")

    st.write("""
This project utilizes **Transfer Learning**, where a pre-trained
MobileNetV2 model (trained on ImageNet) is reused to classify
images of cats and dogs.

Instead of training a CNN from scratch, the model leverages
existing learned visual features, reducing training time while
achieving excellent accuracy.
""")

    st.divider()

    # ======================================================
    # TECHNOLOGIES
    # ======================================================

    st.header("🛠 Technologies Used")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
### Programming

- Python 3.11

- TensorFlow / Keras

- NumPy

- Pandas

- Matplotlib

- Plotly

- Pillow

- Streamlit
""")

    with col2:

        st.markdown("""
### Deep Learning

- MobileNetV2

- Transfer Learning

- Fine Tuning

- Data Augmentation

- Binary Classification

- Image Processing
""")

    st.divider()

    # ======================================================
    # MODEL PIPELINE
    # ======================================================

    st.header("🧠 Model Pipeline")

    st.info("""

Image Upload

⬇

Resize (224 × 224)

⬇

Normalization

⬇

Data Augmentation

⬇

MobileNetV2 Feature Extractor

⬇

Global Average Pooling

⬇

Dense Layer

⬇

Dropout

⬇

Softmax Output

⬇

Cat / Dog Prediction

""")

    st.divider()

    # ======================================================
    # FEATURES
    # ======================================================

    st.header("✨ Application Features")

    features = [

        "✔ Upload Custom Images",

        "✔ AI Predictions",

        "✔ Confidence Score",

        "✔ Probability Distribution",

        "✔ Interactive Charts",

        "✔ Dataset Explorer",

        "✔ Prediction History",

        "✔ Analytics Dashboard",

        "✔ Model Details",

        "✔ Training Curves",

        "✔ Confusion Matrix",

        "✔ Download Prediction Reports",

        "✔ Download Prediction History"

    ]

    for feature in features:

        st.write(feature)

    st.divider()

    # ======================================================
    # PROJECT STATISTICS
    # ======================================================

    st.header("📊 Project Statistics")

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric("Classes", "2")

    with c2:

        st.metric("Input Size", "224×224")

    with c3:

        st.metric("Architecture", "MobileNetV2")

    with c4:

        st.metric("Accuracy", "99.22%")

    st.divider()

    # ======================================================
    # LEARNING OUTCOMES
    # ======================================================

    st.header("🎯 Learning Outcomes")

    st.markdown("""

After completing this project, the following concepts were learned:

- Transfer Learning

- Feature Extraction

- Fine-Tuning

- MobileNetV2

- Data Augmentation

- Binary Image Classification

- TensorFlow/Keras Workflow

- Model Evaluation

- Streamlit Deployment

- Interactive AI Dashboard Development

""")

    st.divider()

    # ======================================================
    # DEVELOPER
    # ======================================================

    st.header("👨‍💻 Developer")

    st.success("""

**Developed By**

Hadeed Jalani

BS Computer Science

University of Lahore

MLBench Summer Internship

AI • Machine Learning • Deep Learning

""")

    st.divider()

    # ======================================================
    # VERSION
    # ======================================================

    st.caption("""
Version 1.0

Powered by TensorFlow • MobileNetV2 • Streamlit

© 2026 MLBench Summer Internship

""")
