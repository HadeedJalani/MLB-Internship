# ==================================================
#        DATA PREPROCESSING
#           MLBench Summer Internship
# ==================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

DATASET = "student_performance.csv"


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

data = pd.read_csv(DATASET)

print("\nDataset Loaded Successfully!")


# --------------------------------------------------
# Encode Program Column
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

print("Features and Target Selected!")

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
# Summary
# --------------------------------------------------

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))