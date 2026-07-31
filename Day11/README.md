# 🌸 Day 11 – Iris Flower Clustering & PCA Visualization

## MLBench Summer Internship

An interactive Machine Learning project demonstrating **Unsupervised Learning** using the **Iris Dataset**. This project applies **K-Means Clustering** to group similar flowers and **Principal Component Analysis (PCA)** for dimensionality reduction, presented through an interactive **Streamlit dashboard**.

---

## 🚀 Live Demo

🌐 **Streamlit App:**  
https://mlb-internship-xpcpthicmu4axlhmgjanag.streamlit.app/

📂 **GitHub Repository:**  
https://github.com/HadeedJalani/MLB-Internship

---

# 📌 Project Objectives

- Explore the Iris Dataset
- Perform K-Means Clustering
- Determine the optimal number of clusters using the Elbow Method
- Apply Principal Component Analysis (PCA)
- Visualize clustering results
- Build an interactive Streamlit dashboard
- Allow users to upload their own datasets

---

# 📂 Project Structure

```text
Day11/
│
├── dataset_exploration.py
├── iris_clustering_visualization.py
├── kmeans_clustering.py
├── pca_analysis.py
│
├── streamlit_app.py
├── requirements.txt
├── README.md
│
├── elbow_method.png
├── kmeans_clusters.png
├── original_data.png
├── pca_visualization.png
└── pca_3d_visualization.png
```

---

# 📊 Dataset Information

| Attribute | Value |
|-----------|-------|
| Dataset | Iris Dataset |
| Source | Scikit-Learn |
| Samples | 150 |
| Features | 4 |
| Classes | Setosa, Versicolor, Virginica |

---

# 🌼 What is K-Means Clustering?

K-Means is one of the most widely used **Unsupervised Machine Learning** algorithms.

It works by:

1. Selecting **K** cluster centroids.
2. Assigning every data point to its nearest centroid.
3. Updating centroid locations.
4. Repeating the process until convergence.

Since the Iris dataset naturally contains three flower species, the optimal number of clusters is approximately **K = 3**.

---

# 📉 Elbow Method

The Elbow Method helps identify the optimal value of **K** by plotting the **Within Cluster Sum of Squares (WCSS)** against different cluster counts.

The "elbow" point represents the best trade-off between model complexity and clustering quality.

---

# 📚 Principal Component Analysis (PCA)

PCA reduces high-dimensional data into fewer dimensions while preserving most of the important information.

### Benefits

- Easier visualization
- Reduced dimensionality
- Faster computation
- Better understanding of data patterns
- Noise reduction

---

# 📈 Visualizations Included

The project generates:

- 📌 Original Dataset Visualization
- 📌 Elbow Method Curve
- 📌 K-Means Cluster Plot
- 📌 PCA 2D Visualization
- 📌 Interactive PCA 3D Visualization

---

# 🌐 Streamlit Dashboard Features

The application allows users to:

- Upload custom CSV datasets
- Preview uploaded data
- Choose the number of clusters (K)
- Perform K-Means clustering
- Visualize cluster distribution
- Generate interactive 2D PCA plots
- Explore interactive 3D PCA visualizations
- Download clustered data as CSV
- View cluster centroids

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Plotly
- Streamlit

---

# 📌 Key Observations

- The Elbow Method suggests **K = 3** as the optimal number of clusters for the Iris dataset.
- K-Means effectively groups flowers with similar characteristics.
- PCA reduces four dimensions into two principal components while preserving most of the dataset's information.
- Interactive visualizations improve understanding of cluster separation and data distribution.

---

# ▶️ Getting Started

## 1. Clone the Repository

```bash
git clone https://github.com/HadeedJalani/MLB-Internship.git
```

## 2. Navigate to Day11

```bash
cd MLB-Internship/Day11
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the Streamlit Application

```bash
streamlit run streamlit_app.py
```

---

# 📸 Project Outputs

The project automatically generates:

- ✔️ Original Data Visualization
- ✔️ Elbow Method Plot
- ✔️ K-Means Cluster Plot
- ✔️ PCA 2D Visualization
- ✔️ PCA 3D Visualization

---

# 🔗 Project Links

### 🌐 Live Streamlit Application

https://mlb-internship-xpcpthicmu4axlhmgjanag.streamlit.app/

### 💻 GitHub Repository

https://github.com/HadeedJalani/MLB-Internship

---

# 👨‍💻 Developed By

**Hadeed Jalani**

BS Computer Science — University of Lahore

MLBench Summer Internship

Day 11 – Iris Flower Clustering & PCA Visualization
