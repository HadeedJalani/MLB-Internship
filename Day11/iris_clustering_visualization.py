# ==========================================================
# MLBench Summer Internship - Day 11
#
# Mini Project:
# Iris Flower Clustering & Visualization
#
# Author: Hadeed Jalani
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

data = pd.DataFrame(
    X,
    columns=feature_names,
)

data["Species"] = y

print("=" * 70)
print("IRIS FLOWER CLUSTERING & VISUALIZATION")
print("=" * 70)

print("\nDataset Shape")
print(data.shape)

print("\nFirst Five Records")
print(data.head())

print("\nStatistical Summary")
print(data.describe())

print("\nTarget Distribution")
print(data["Species"].value_counts())

print("\nTarget Labels")

for index, name in enumerate(target_names):

    print(f"{index} -> {name}")
# --------------------------------------------------
# Original Data Visualization
# --------------------------------------------------

print("\nGenerating Original Data Visualization...")

colors = [
    "red",
    "green",
    "blue",
]

plt.figure(figsize=(8, 6))

for i, species in enumerate(target_names):

    plt.scatter(
        data[data["Species"] == i]["sepal length (cm)"],
        data[data["Species"] == i]["petal length (cm)"],
        color=colors[i],
        label=species,
        s=60,
    )

plt.title("Original Iris Dataset")

plt.xlabel("Sepal Length (cm)")

plt.ylabel("Petal Length (cm)")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "original_data.png",
    dpi=300,
)

plt.close()

print("Original Data graph saved as original_data.png")
# --------------------------------------------------
# Elbow Method
# --------------------------------------------------

print("\nCalculating WCSS for different values of K...")

wcss = []

for k in range(1, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
    )

    model.fit(X)

    wcss.append(model.inertia_)

print("\nWCSS Values")

for i, value in enumerate(wcss, start=1):

    print(f"K = {i}  -->  WCSS = {value:.2f}")

# --------------------------------------------------
# Plot Elbow Method
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.plot(
    range(1, 11),
    wcss,
    marker="o",
    linewidth=2,
)

plt.title("Elbow Method")

plt.xlabel("Number of Clusters (K)")

plt.ylabel("WCSS")

plt.xticks(range(1, 11))

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "elbow_method.png",
    dpi=300,
)

plt.close()

print("\nElbow Method graph saved as elbow_method.png")

print("\nObservation:")
print("The elbow appears around K = 3.")
print("Therefore, the optimal number of clusters is 3.")    

# --------------------------------------------------
# Apply K-Means Clustering
# --------------------------------------------------

print("\nApplying K-Means Clustering...")

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10,
)

clusters = kmeans.fit_predict(X)

data["Cluster"] = clusters

# --------------------------------------------------
# Cluster Information
# --------------------------------------------------

print("\nCluster Centers")

centers = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=feature_names,
)

print(centers)

print("\nCluster Distribution")

print(data["Cluster"].value_counts().sort_index())

# --------------------------------------------------
# Cluster Visualization
# --------------------------------------------------

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    data["sepal length (cm)"],
    data["petal length (cm)"],
    c=data["Cluster"],
    cmap="viridis",
    s=60,
)

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 2],
    color="red",
    marker="X",
    s=220,
    label="Centroids",
)

plt.title("K-Means Clustering on Iris Dataset")

plt.xlabel("Sepal Length (cm)")

plt.ylabel("Petal Length (cm)")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "kmeans_clusters.png",
    dpi=300,
)

plt.close()

print("\nCluster visualization saved as kmeans_clusters.png")

# --------------------------------------------------
# Compare Actual Species with Clusters
# --------------------------------------------------

print("\nActual Species vs Cluster Labels")

comparison = pd.crosstab(
    data["Species"],
    data["Cluster"],
)

print(comparison)
# --------------------------------------------------
# Principal Component Analysis (PCA)
# --------------------------------------------------

print("\nApplying PCA...")

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

print("\nOriginal Shape")

print(X.shape)

print("\nReduced Shape")

print(X_pca.shape)

print("\nExplained Variance Ratio")

print(pca.explained_variance_ratio_)

print(
    f"\nTotal Variance Retained : "
    f"{sum(pca.explained_variance_ratio_) * 100:.2f}%"
)

# --------------------------------------------------
# Create PCA DataFrame
# --------------------------------------------------

pca_df = pd.DataFrame(
    X_pca,
    columns=[
        "Principal Component 1",
        "Principal Component 2",
    ],
)

pca_df["Species"] = y

# --------------------------------------------------
# 2D PCA Visualization
# --------------------------------------------------

plt.figure(figsize=(8, 6))

for i, species in enumerate(target_names):

    plt.scatter(
        pca_df[pca_df["Species"] == i]["Principal Component 1"],
        pca_df[pca_df["Species"] == i]["Principal Component 2"],
        label=species,
        s=60,
    )

plt.title("PCA Visualization of Iris Dataset")

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "pca_visualization.png",
    dpi=300,
)

plt.close()

print("\nPCA visualization saved as pca_visualization.png")

# --------------------------------------------------
# 3D PCA Visualization
# --------------------------------------------------

pca_3d = PCA(n_components=3)

X_pca_3d = pca_3d.fit_transform(X)

fig = plt.figure(figsize=(9, 7))

ax = fig.add_subplot(111, projection="3d")

for i, species in enumerate(target_names):

    ax.scatter(
        X_pca_3d[y == i, 0],
        X_pca_3d[y == i, 1],
        X_pca_3d[y == i, 2],
        label=species,
        s=50,
    )

ax.set_title("3D PCA Visualization")

ax.set_xlabel("PC1")

ax.set_ylabel("PC2")

ax.set_zlabel("PC3")

ax.legend()

plt.tight_layout()

plt.savefig(
    "pca_3d_visualization.png",
    dpi=300,
)

plt.close()

print("3D PCA visualization saved as pca_3d_visualization.png")

# --------------------------------------------------
# Observations
# --------------------------------------------------

print("\n" + "=" * 70)
print("OBSERVATIONS")
print("=" * 70)

print("\n1. The Elbow Method indicates K = 3 as the optimal number of clusters.")

print("\n2. K-Means successfully grouped the Iris dataset into three clusters.")

print("\n3. Setosa is clearly separated from the other species.")

print("\n4. Versicolor and Virginica show slight overlap.")

print("\n5. PCA reduced the dataset from 4 dimensions to 2 dimensions.")

print(
    f"\n6. PCA retained "
    f"{sum(pca.explained_variance_ratio_) * 100:.2f}% "
    f"of the original information."
)

print("\n7. PCA made cluster visualization much easier.")

print("\nScript Completed Successfully.")