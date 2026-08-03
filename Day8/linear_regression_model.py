# ==================================================
#        LINEAR REGRESSION MODEL
#          MLBench Summer Internship
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

DATASET = "student_performance.csv"

data = pd.read_csv(DATASET)

print("\nDataset Loaded Successfully!")


# --------------------------------------------------
# Encode Categorical Columns
# --------------------------------------------------

encoder = LabelEncoder()

data["Program"] = encoder.fit_transform(data["Program"])

print("Categorical Data Encoded!")

# --------------------------------------------------
# Features and Target
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
# Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.20,
    random_state=42

)

print("Dataset Split Successfully!")

# --------------------------------------------------
# Feature Scaling
# --------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

print("Feature Scaling Applied!")

# --------------------------------------------------
# Train Linear Regression Model
# --------------------------------------------------

model = LinearRegression()

model.fit(X_train, y_train)

print("Linear Regression Model Trained!")

# --------------------------------------------------
# Predictions
# --------------------------------------------------

predictions = model.predict(X_test)

print("\nActual vs Predicted Python Marks")
print("-" * 45)

comparison = pd.DataFrame({

    "Actual Python Marks": y_test.values,
    "Predicted Python Marks": predictions

})

print(comparison)

# --------------------------------------------------
# Evaluation Metrics
# --------------------------------------------------

mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

r2 = r2_score(y_test, predictions)

print("\nModel Evaluation")
print("-" * 45)

print(f"Mean Absolute Error : {mae:.2f}")
print(f"Mean Squared Error  : {mse:.2f}")
print(f"R² Score            : {r2:.2f}")

# --------------------------------------------------
# Scatter Plot
# --------------------------------------------------

plt.figure(figsize=(8,6))

plt.scatter(y_test, predictions)

plt.xlabel("Actual Average Score")

plt.ylabel("Predicted Average Score")

plt.title("Actual vs Predicted Scores")

plt.tight_layout()

plt.savefig("prediction_scatter.png")

plt.show()

print("\nScatter Plot Saved Successfully!")