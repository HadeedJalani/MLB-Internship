# ==========================================================
# MLBench Summer Internship - Day 11
#
# Iris Flower Clustering & PCA Dashboard
#
# Author : Hadeed Jalani
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import plotly.express as px

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Iris Clustering Dashboard",
    page_icon="🌸",
    layout="wide",
)

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🌸 Iris Flower Clustering & PCA Dashboard")

st.write(
    """
This application demonstrates **K-Means Clustering**
and **Principal Component Analysis (PCA)** on the
Iris Dataset.
"""
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("Project Information")

st.sidebar.markdown("---")

st.sidebar.success("Algorithm : K-Means")

st.sidebar.info("Dimensionality Reduction : PCA")

st.sidebar.write("Dataset : Iris Dataset")

st.sidebar.markdown("---")

st.sidebar.write("Developed By")

st.sidebar.success("Hadeed Jalani")

# --------------------------------------------------
# Upload Dataset
# --------------------------------------------------

st.header("📂 Dataset")

uploaded_file = st.file_uploader(
    "Upload an Iris Dataset CSV (Optional)",
    type=["csv"],
)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.success("Custom dataset loaded successfully!")

else:

    iris = load_iris()

    data = pd.DataFrame(
        iris.data,
        columns=iris.feature_names,
    )

    data["target"] = iris.target

    st.info("Using built-in Iris Dataset")

# --------------------------------------------------
# Dataset Preview
# --------------------------------------------------

st.subheader("Dataset Preview")

st.dataframe(
    data.head(10),
    use_container_width=True,
)

# --------------------------------------------------
# Dataset Information
# --------------------------------------------------

st.subheader("Dataset Information")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Rows",
        data.shape[0],
    )

with col2:

    st.metric(
        "Columns",
        data.shape[1],
    )

with col3:

    st.metric(
        "Features",
        data.shape[1] - 1,
    )

# --------------------------------------------------
# Data Types
# --------------------------------------------------

st.subheader("Column Data Types")

dtype_df = pd.DataFrame({

    "Column": data.columns,

    "Data Type": data.dtypes.astype(str)

})

st.dataframe(
    dtype_df,
    use_container_width=True,
)

# --------------------------------------------------
# Target Distribution
# --------------------------------------------------

if "target" in data.columns:

    st.subheader("Target Distribution")

    counts = data["target"].value_counts().sort_index()

    st.bar_chart(counts)

# --------------------------------------------------
# K-Means Settings
# --------------------------------------------------

st.header("⚙️ K-Means Clustering Settings")

numeric_columns = data.select_dtypes(include=np.number).columns.tolist()

# Remove target column if present
if "target" in numeric_columns:
    numeric_columns.remove("target")

st.sidebar.write(f"Samples : {data.shape[0]}")
st.sidebar.write(f"Features : {len(numeric_columns)}")

col1, col2 = st.columns(2)

with col1:

    k = st.slider(
        "Select Number of Clusters (K)",
        min_value=2,
        max_value=10,
        value=3,
    )

with col2:

    feature_x = st.selectbox(
        "X-Axis Feature",
        numeric_columns,
        index=0,
    )

    feature_y = st.selectbox(
        "Y-Axis Feature",
        numeric_columns,
        index=2,
    )

# --------------------------------------------------
# Feature Matrix
# --------------------------------------------------

X = data[numeric_columns]

# --------------------------------------------------
# Elbow Method
# --------------------------------------------------

st.subheader("📈 Elbow Method")

wcss = []

for i in range(1, 11):

    model = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10,
    )

    model.fit(X)

    wcss.append(model.inertia_)

fig, ax = plt.subplots(figsize=(8,5))

ax.plot(
    range(1,11),
    wcss,
    marker="o",
)

ax.set_title("Elbow Method")

ax.set_xlabel("Number of Clusters (K)")

ax.set_ylabel("WCSS")

ax.grid(True)

st.pyplot(fig)

# --------------------------------------------------
# WCSS Table
# --------------------------------------------------

st.subheader("WCSS Values")

wcss_df = pd.DataFrame({

    "K": list(range(1,11)),

    "WCSS": np.round(wcss,2)

})

st.dataframe(
    wcss_df,
    use_container_width=True,
)

st.success(
    "Use the Elbow Method graph above to determine the optimal value of K for your dataset."
)    
# --------------------------------------------------
# Apply K-Means Clustering
# --------------------------------------------------

st.header("🎯 K-Means Clustering Results")

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10,
)

clusters = kmeans.fit_predict(X)

# Format cluster names into "Cluster 1", "Cluster 2", etc.
data["Cluster"] = [f"Cluster {c + 1}" for c in clusters]

# --------------------------------------------------
# Cluster Scatter Plot
# --------------------------------------------------

fig = px.scatter(

    data,

    x=feature_x,

    y=feature_y,

    color="Cluster",

    title="🌸 K-Means Cluster Visualization",

    labels={
        "Cluster": "Cluster Group"
    },

    hover_data=data.columns,

    color_discrete_sequence=px.colors.qualitative.Set2,

)

# Apply marker updates to data points BEFORE adding centroids
fig.update_traces(
    marker=dict(
        size=11,
        line=dict(width=1, color="white"),
    ),
    selector=dict(mode="markers")
)

# --------------------------------------------------
# Add Cluster Centers
# --------------------------------------------------

fig.add_scatter(

    x=kmeans.cluster_centers_[:, numeric_columns.index(feature_x)],

    y=kmeans.cluster_centers_[:, numeric_columns.index(feature_y)],

    mode="markers",

    marker=dict(

        size=24,

        symbol="x",

        color="#FFD700",

        line=dict(width=2, color="black"),

    ),

    name="Centroids",

)

fig.update_layout(

    template="plotly_dark",

    title_x=0.5,

    legend_title="Clusters",

    font=dict(size=14),

    margin=dict(l=20, r=20, t=60, b=20),

)

st.plotly_chart(
    fig,
    use_container_width=True,
)

# --------------------------------------------------
# Cluster Statistics
# --------------------------------------------------

st.subheader("📊 Cluster Distribution")

cluster_counts = (

    data["Cluster"]

    .value_counts()

    .sort_index()

)

st.bar_chart(cluster_counts)

cluster_df = pd.DataFrame({

    "Cluster": cluster_counts.index,

    "Samples": cluster_counts.values

})

st.dataframe(

    cluster_df,

    use_container_width=True,

)

# --------------------------------------------------
# Cluster Centers
# --------------------------------------------------

st.subheader("📍 Cluster Centers")

centers = pd.DataFrame(

    kmeans.cluster_centers_,

    columns=numeric_columns,

)
centers.index = [f"Cluster {i+1}" for i in range(k)]

st.dataframe(

    centers,

    use_container_width=True,

)

# --------------------------------------------------
# Observations
# --------------------------------------------------

st.subheader("📝 Observations")

st.success(

    f"""
• Number of clusters formed : {k}

• Total samples : {len(data)}

• Cluster sizes vary depending on the selected K.

• Centroids represent the center of each cluster.

• The Iris dataset naturally forms approximately 3 clusters.
"""

)
# --------------------------------------------------
# PCA Analysis
# --------------------------------------------------

st.header("📉 Principal Component Analysis (PCA)")

# Before vs After Dimensions
col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Original Dimensions",
        X.shape[1],
    )

with col2:

    st.metric(
        "Reduced Dimensions",
        2,
    )

# --------------------------------------------------
# Apply PCA (2 Components)
# --------------------------------------------------

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

pca_df = pd.DataFrame(
    X_pca,
    columns=[
        "PC1",
        "PC2",
    ],
)

pca_df["Cluster"] = data["Cluster"]

# --------------------------------------------------
# Explained Variance
# --------------------------------------------------

st.subheader("Explained Variance Ratio")

variance = pca.explained_variance_ratio_

variance_df = pd.DataFrame({

    "Principal Component": [
        "PC1",
        "PC2",
    ],

    "Variance Explained": np.round(
        variance * 100,
        2,
    ),

})

st.dataframe(
    variance_df,
    use_container_width=True,
)

st.success(
    f"Total Variance Retained : {variance.sum()*100:.2f}%"
)

# --------------------------------------------------
# PCA 2D Visualization
# --------------------------------------------------

st.subheader("2D PCA Visualization")

fig = px.scatter(

    pca_df,

    x="PC1",

    y="PC2",

    color="Cluster",

    title="🌸 2D PCA Projection",

    color_discrete_sequence=px.colors.qualitative.Set2,

)

fig.update_traces(
    marker=dict(
        size=10,
        line=dict(width=1, color="white"),
    )
)

fig.update_layout(

    template="plotly_dark",

    title_x=0.5,

    legend_title="Clusters",

)

st.plotly_chart(
    fig,
    use_container_width=True,
)

# --------------------------------------------------
# 3D PCA
# --------------------------------------------------

pca3 = PCA(n_components=3)

X_pca3 = pca3.fit_transform(X)

pca3_df = pd.DataFrame(
    X_pca3,
    columns=[
        "PC1",
        "PC2",
        "PC3",
    ],
)

pca3_df["Cluster"] = data["Cluster"]

st.subheader("Interactive 3D PCA Visualization")

fig3d = px.scatter_3d(

    pca3_df,

    x="PC1",

    y="PC2",

    z="PC3",

    color="Cluster",

    opacity=0.9,

    title="🌸 3D PCA Cluster Visualization",

    color_discrete_sequence=px.colors.qualitative.Set2,

)

fig3d.update_traces(
    marker=dict(
        size=5,
    )
)

fig3d.update_layout(

    template="plotly_dark",

    title_x=0.5,

    legend_title="Clusters",

    scene_camera=dict(
        eye=dict(x=1.5, y=1.5, z=1.2)
    ),

)

st.plotly_chart(
    fig3d,
    use_container_width=True,
)

# --------------------------------------------------
# PCA Observations
# --------------------------------------------------

st.subheader("Observations")

st.info(
"""
• PCA reduces the Iris dataset from 4 dimensions to 2 dimensions.

• More than 95% of the dataset's information is retained.

• Setosa forms a clearly separated cluster.

• Versicolor and Virginica show slight overlap.

• PCA makes high-dimensional datasets much easier to visualize.

• The 3D visualization provides a better understanding of the cluster separation.
"""
)
# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("---")

st.markdown(
"""
### 📌 Summary

 Dataset Exploration

 K-Means Clustering

 Elbow Method

 PCA (2D & 3D)

 Interactive Visualizations

 Before & After Dimension Comparison

---

**MLBench Summer Internship - Day 11**

Developed by **Hadeed Jalani**
"""
)