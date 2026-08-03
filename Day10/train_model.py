# ==================================================
#        TRAIN FINAL MODEL
#        MLBench Summer Internship - Day 10
# ==================================================

import joblib
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

print("=" * 70)
print("Loading Breast Cancer Dataset...")
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
# Train Final Model
# --------------------------------------------------

model = LogisticRegression(

    C=1,

    solver="liblinear",

    max_iter=1000,

    random_state=42,

)

model.fit(
    X_train,
    y_train,
)

# --------------------------------------------------
# Save Model & Scaler
# --------------------------------------------------

joblib.dump(
    model,
    "logistic_regression_model.pkl",
)

joblib.dump(
    scaler,
    "scaler.pkl",
)

print("\nModel Saved Successfully!")

print("logistic_regression_model.pkl")

print("scaler.pkl")