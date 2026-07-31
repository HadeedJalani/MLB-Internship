
# ==========================================================
# MLBench Summer Internship - Day 11
#
# Project:
# K-Means Clustering on Iris Dataset
#
# Author: Hadeed Jalani
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

iris = load_iris()

data = pd.DataFrame(
    iris.data,
    columns=iris.feature_names,
)

print("=" * 70)
print("IRIS DATASET LOADED")
print("=" * 70)

print(f"Dataset Shape : {data.shape}")

# --------------------------------------------------
# Elbow Method
# --------------------------------------------------

print("\nCalculating WCSS values...")

wcss = []

for k in range(1, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
    )

    model.fit(data)

    wcss.append(model.inertia_)

# --------------------------------------------------
# Plot Elbow Method
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 11),
    wcss,
    marker="o",
)

plt.title("Elbow Method")

plt.xlabel("Number of Clusters (K)")

plt.ylabel("WCSS")

plt.grid(True)

plt.savefig("elbow_method.png")

print("Elbow Method graph saved as elbow_method.png")

# --------------------------------------------------
# Apply KMeans
# --------------------------------------------------

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10,
)

clusters = kmeans.fit_predict(data)

data["Cluster"] = clusters

print("\nCluster Centers\n")

print(kmeans.cluster_centers_)

print("\nCluster Distribution\n")

print(data["Cluster"].value_counts())

# --------------------------------------------------
# Scatter Plot
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    data["sepal length (cm)"],
    data["petal length (cm)"],
    c=data["Cluster"],
)

plt.xlabel("Sepal Length (cm)")

plt.ylabel("Petal Length (cm)")

plt.title("K-Means Clustering")

plt.savefig("kmeans_clusters.png")

print("\nCluster graph saved as kmeans_clusters.png")

print("\nScript Completed Successfully.")