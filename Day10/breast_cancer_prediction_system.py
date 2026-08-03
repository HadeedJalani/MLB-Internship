# ==================================================
#     BREAST CANCER PREDICTION SYSTEM
#        MLBench Summer Internship - Day 10
# ==================================================

import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer

# --------------------------------------------------
# Load Saved Model and Scaler
# --------------------------------------------------

model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

# --------------------------------------------------
# Load Dataset Information
# --------------------------------------------------

data = load_breast_cancer()

feature_names = data.feature_names

classes = {
    0: "Malignant",
    1: "Benign",
}

# --------------------------------------------------
# Header
# --------------------------------------------------

print("=" * 70)
print("BREAST CANCER PREDICTION SYSTEM")
print("=" * 70)

print("\nEnter the feature values below.")
print("Type 'q' anytime to quit.\n")

# --------------------------------------------------
# Take User Input
# --------------------------------------------------

values = []

for feature in feature_names:

    while True:

        user_input = input(f"{feature}: ")

        if user_input.lower() == "q":

            print("\nPrediction cancelled.")
            exit()

        if user_input.strip() == "":

            print("Input cannot be empty. Please enter a value.\n")
            continue

        try:

            value = float(user_input)

            values.append(value)

            break

        except ValueError:

            print("Invalid number. Please enter a numeric value.\n")

# --------------------------------------------------
# Convert to NumPy Array
# --------------------------------------------------

sample = np.array(values).reshape(1, -1)

# --------------------------------------------------
# Scale Features
# --------------------------------------------------

sample = scaler.transform(sample)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

prediction = model.predict(sample)[0]

probability = model.predict_proba(sample)[0]

confidence = np.max(probability) * 100

# --------------------------------------------------
# Display Results
# --------------------------------------------------

print("\n")
print("=" * 70)
print("PREDICTION RESULT")
print("=" * 70)

print(f"\nPredicted Class : {classes[prediction]}")
print(f"Confidence      : {confidence:.2f}%")

print("\nProbability Distribution")
print("-" * 70)
print(f"Malignant : {probability[0] * 100:.2f}%")
print(f"Benign    : {probability[1] * 100:.2f}%")

print("\n")

if prediction == 1:

    print("Interpretation:")
    print("The model predicts that the tumor is likely BENIGN.")

else:

    print("Interpretation:")
    print("The model predicts that the tumor is likely MALIGNANT.")

print("\n")
print("=" * 70)
print("Prediction Completed Successfully!")
print("=" * 70)