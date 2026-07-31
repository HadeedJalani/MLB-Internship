# ==========================================================
# MLBench Summer Internship - Day 11
#
# Project:
# PCA Visualization on Iris Dataset
#
# Author: Hadeed Jalani
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

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
    columns=feature_names
)

print("=" * 70)
print("IRIS DATASET LOADED")
print("=" * 70)

print(f"Original Shape : {X.shape}")

# --------------------------------------------------
# Apply PCA
# --------------------------------------------------

print("\nApplying PCA...\n")

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

print("Reduced Shape")

print(X_pca.shape)

print("\nExplained Variance Ratio")

print(pca.explained_variance_ratio_)

print(
    f"\nTotal Variance Retained : "
    f"{sum(pca.explained_variance_ratio_)*100:.2f}%"
)

# --------------------------------------------------
# Create DataFrame
# --------------------------------------------------

pca_df = pd.DataFrame(
    X_pca,
    columns=[
        "Principal Component 1",
        "Principal Component 2",
    ]
)

pca_df["Species"] = y

# --------------------------------------------------
# Plot PCA
# --------------------------------------------------

colors = [
    "red",
    "green",
    "blue",
]

plt.figure(figsize=(8, 6))

for i, species in enumerate(target_names):

    plt.scatter(
        pca_df[pca_df["Species"] == i]["Principal Component 1"],
        pca_df[pca_df["Species"] == i]["Principal Component 2"],
        label=species,
        color=colors[i],
    )

plt.title("PCA Visualization of Iris Dataset")

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.legend()

plt.grid(True)

plt.savefig(
    "pca_visualization.png",
    dpi=300,
    bbox_inches="tight",
)

print("\nPCA graph saved as pca_visualization.png")

print("\nScript Completed Successfully.")