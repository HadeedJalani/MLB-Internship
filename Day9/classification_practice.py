# ==========================================================
# MLBench Summer Internship - Day 9
# Classification Practice using Logistic Regression
#
# Topics Covered are as follows on Day 9:
# - Classification Workflow
# - Iris Dataset
# - Feature Exploration
# - Target Classes
# - Logistic Regression
# - Model Prediction
# - Model Evaluation
# ==========================================================

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split


def load_dataset():
    """Load the Iris dataset."""

    iris = load_iris()

    return iris


def explore_dataset(iris):
    """Display dataset information."""

    print("=" * 70)
    print("IRIS DATASET EXPLORATION")
    print("=" * 70)

    print(f"Dataset Shape       : {iris.data.shape}")
    print(f"Total Samples       : {len(iris.data)}")
    print(f"Total Features      : {len(iris.feature_names)}")
    print(f"Total Classes       : {len(iris.target_names)}")

    print("\nFeature Names:")
    for feature in iris.feature_names:
        print(f"• {feature}")

    print("\nTarget Classes:")
    for target in iris.target_names:
        print(f"• {target}")

    print("\nFirst Five Samples:")
    for i in range(5):
        print(
            f"Features: {iris.data[i]} "
            f"| Target: {iris.target_names[iris.target[i]]}"
        )


def split_data(X, y):
    """Split dataset."""

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )


def train_model(X_train, y_train):
    """Train Logistic Regression model."""

    model = LogisticRegression(max_iter=200)

    model.fit(X_train, y_train)

    return model


def make_predictions(model, X_test):
    """Predict flower species."""

    return model.predict(X_test)


def evaluate_model(y_test, predictions):
    """Display evaluation metrics."""

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
    )

    print("\n")
    print("=" * 70)
    print("MODEL PERFORMANCE")
    print("=" * 70)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")


def display_predictions(predictions, y_test, iris):
    """Display sample predictions."""

    print("\n")
    print("=" * 70)
    print("SAMPLE PREDICTIONS")
    print("=" * 70)

    for i in range(10):

        actual = iris.target_names[y_test[i]]
        predicted = iris.target_names[predictions[i]]

        status = "✓ Correct" if actual == predicted else "✗ Incorrect"

        print(
            f"Sample {i + 1}\n"
            f"Actual    : {actual}\n"
            f"Predicted : {predicted}\n"
            f"Result    : {status}\n"
        )


def main():

    iris = load_dataset()

    explore_dataset(iris)

    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)

    predictions = make_predictions(model, X_test)

    evaluate_model(y_test, predictions)

    display_predictions(predictions, y_test, iris)


if __name__ == "__main__":
    main()