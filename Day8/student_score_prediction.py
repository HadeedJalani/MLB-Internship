# ==================================================
#      STUDENT SCORE PREDICTION SYSTEM
#        MLBench Summer Internship
# ==================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

print("\nLoading Dataset...")

data = pd.read_csv("student_performance.csv")

print("Dataset Loaded Successfully!")


# --------------------------------------------------
# Encode Categorical Data
# --------------------------------------------------

encoder = LabelEncoder()

data["Program"] = encoder.fit_transform(data["Program"])

print("Categorical Columns Encoded!")

# --------------------------------------------------
# Features & Target
# --------------------------------------------------

X = data[
[
"Age",
"Program",
"Mathematics",
"Statistics",
"Machine_Learning",
"Attendance"
]
]

y = data["Python"]

# --------------------------------------------------
# Train Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Train-Test Split Completed!")

# --------------------------------------------------
# Feature Scaling
# --------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

print("Feature Scaling Completed!")

# --------------------------------------------------
# Train Model
# --------------------------------------------------

model = LinearRegression()

model.fit(X_train, y_train)

print("Model Training Completed!")

# --------------------------------------------------
# Predictions
# --------------------------------------------------

predictions = model.predict(X_test)

comparison = pd.DataFrame({

    "Actual Python Marks": y_test.values,
    "Predicted Python Marks": predictions

})

print("\n")
print("="*60)
print("ACTUAL  VS PREDICTED")
print("="*60)

print(comparison)

# --------------------------------------------------
# Metrics
# --------------------------------------------------

mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

r2 = r2_score(y_test, predictions)

print("\n")
print("="*60)
print("MODEL EVALUATION")
print("="*60)

print(f"Mean Absolute Error : {mae:.2f}")
print(f"Mean Squared Error  : {mse:.2f}")
print(f"R² Score            : {r2:.2f}")

# --------------------------------------------------
# Visualization
# --------------------------------------------------

plt.figure(figsize=(8,6))

plt.scatter(y_test, predictions)

plt.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    "r--"
)

plt.xlabel("Actual Score")

plt.ylabel("Predicted Score")

plt.title("Actual vs Predicted Scores")

plt.tight_layout()

plt.savefig("prediction_scatter.png")

plt.show()

print("\nScatter Plot Saved Successfully!")

print("\nStudent Score Prediction System Completed Successfully!")