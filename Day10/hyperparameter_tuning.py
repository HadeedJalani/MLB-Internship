# ==================================================
#     HYPERPARAMETER TUNING USING GRIDSEARCHCV
#        MLBench Summer Internship - Day 10
# ==================================================

import pandas as pd
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
)

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
    test_size=0.20,
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
# Base Model
# --------------------------------------------------

model = LogisticRegression(
    random_state=42,
    max_iter=1000,
)

# --------------------------------------------------
# Hyperparameter Grid
# --------------------------------------------------

param_grid = {

    "C": [
        0.01,
        0.1,
        1,
        10,
        100,
    ],

    "solver": [
        "liblinear",
        "lbfgs",
    ],

}

# --------------------------------------------------
# Grid Search
# --------------------------------------------------

print("\nPerforming Grid Search...\n")

grid_search = GridSearchCV(

    estimator=model,

    param_grid=param_grid,

    cv=5,

    scoring="accuracy",

    n_jobs=-1,

)

grid_search.fit(
    X_train,
    y_train,
)

# --------------------------------------------------
# Best Model
# --------------------------------------------------

best_model = grid_search.best_estimator_

predictions = best_model.predict(
    X_test,
)

# --------------------------------------------------
# Evaluation
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
print("GRID SEARCH RESULTS")
print("=" * 70)

print()

print("Best Parameters")

print(grid_search.best_params_)

print()

print(f"Best Cross Validation Score : {grid_search.best_score_:.4f}")

print()

print("=" * 70)
print("TUNED MODEL RESULTS")
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

print("\nConfusion Matrix\n")

print(cm)

# --------------------------------------------------
# Save Model
# --------------------------------------------------

joblib.dump(
    best_model,
    "logistic_regression_model.pkl",
)

print("\nModel Saved Successfully!")

# --------------------------------------------------
# Plot Confusion Matrix
# --------------------------------------------------

plt.figure(figsize=(6,5))

plt.imshow(
    cm,
    interpolation="nearest",
)

plt.title("Tuned Confusion Matrix")

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
    "confusion_matrix_tuned.png",
    dpi=300,
)

plt.show()

print("\nConfusion Matrix saved as confusion_matrix_tuned.png")