import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET = os.path.join(BASE_DIR, "student_performance.csv")


def load_dataset():
    return pd.read_csv(DATASET)


def preprocess_data():

    data = load_dataset()

    # Encode categorical column
    encoder = LabelEncoder()
    data["Program"] = encoder.fit_transform(data["Program"])

    # Features
    X = data[
        [
            "Age",
            "Program",
            "Mathematics",
            "Statistics",
            "Machine_Learning",
            "Attendance",
        ]
    ]

    # Target
    y = data["Python"]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    # Feature Scaling
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, data


def train_model():

    X_train, X_test, y_train, y_test, data = preprocess_data()

    model = LinearRegression()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    comparison = pd.DataFrame(
        {
            "Actual Python Marks": y_test.values,
            "Predicted Python Marks": predictions,
        }
    )

    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "MSE": mean_squared_error(y_test, predictions),
        "R2": r2_score(y_test, predictions),
    }

    return data, comparison, metrics