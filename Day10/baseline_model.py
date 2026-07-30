# ==================================================
#        BASELINE LOGISTIC REGRESSION MODEL
#        MLBench Summer Internship - Day 10
# ==================================================

import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import matplotlib.pyplot as plt

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

print("=" * 70)
print("Loading Dataset...")
print("=" * 70)

data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names,
)

y = data.target

# --------------------------------------------------
# Split Dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# --------------------------------------------------
# Feature Scaling
# --------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# --------------------------------------------------
# Train Model
# --------------------------------------------------

print("\nTraining Logistic Regression...\n")

model = LogisticRegression(
    max_iter=1000,
    random_state=42,
)

model.fit(
    X_train,
    y_train,
)

# --------------------------------------------------
# Predictions
# --------------------------------------------------

predictions = model.predict(X_test)

# --------------------------------------------------
# Evaluation Metrics
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions,
)

precision = precision_score(
    y_test,
    predictions,
)

recall = recall_score(
    y_test,
    predictions,
)

f1 = f1_score(
    y_test,
    predictions,
)

print("=" * 70)
print("BASELINE MODEL RESULTS")
print("=" * 70)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        predictions,
        target_names=data.target_names,
    )
)

# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    predictions,
)

print("Confusion Matrix\n")

print(cm)

# --------------------------------------------------
# Plot Confusion Matrix
# --------------------------------------------------

plt.figure(figsize=(6,5))

plt.imshow(
    cm,
    interpolation="nearest",
)

plt.title("Baseline Confusion Matrix")

plt.colorbar()

classes = data.target_names

plt.xticks(
    [0,1],
    classes,
)

plt.yticks(
    [0,1],
    classes,
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=12,
        )

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "confusion_matrix_baseline.png",
    dpi=300,
)

plt.show()

print("\nConfusion Matrix saved as:")
print("confusion_matrix_baseline.png")