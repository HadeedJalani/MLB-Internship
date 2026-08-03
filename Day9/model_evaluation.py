# ==========================================================
# MLBench Summer Internship - Day 9
# Model Evaluation using Logistic Regression
#
# Topics Covered:
# - Train/Test Split
# - Logistic Regression
# - Accuracy
# - Precision
# - Recall
# - F1-Score
# - Confusion Matrix
# ==========================================================

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split


def load_dataset():
    """
    Load the Iris dataset.
    """
    iris = load_iris()

    X = iris.data
    y = iris.target

    return X, y, iris


def explore_dataset(iris):
    """
    Display basic information about the dataset.
    """

    print("=" * 60)
    print("IRIS DATASET INFORMATION")
    print("=" * 60)

    print(f"Dataset Shape       : {iris.data.shape}")
    print(f"Total Samples       : {len(iris.data)}")
    print(f"Number of Features  : {len(iris.feature_names)}")
    print(f"Number of Classes   : {len(iris.target_names)}")

    print("\nFeature Names:")
    for feature in iris.feature_names:
        print(f"- {feature}")

    print("\nTarget Classes:")
    for target in iris.target_names:
        print(f"- {target}")

    print("=" * 60)


def split_dataset(X, y):
    """
    Split dataset into training and testing data.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nDataset Split")
    print("-" * 40)
    print(f"Training Samples : {len(X_train)}")
    print(f"Testing Samples  : {len(X_test)}")

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Train Logistic Regression model.
    """

    model = LogisticRegression(max_iter=200)

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate the trained model.
    """

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    precision = precision_score(
        y_test,
        test_predictions,
        average="weighted",
    )

    recall = recall_score(
        y_test,
        test_predictions,
        average="weighted",
    )

    f1 = f1_score(
        y_test,
        test_predictions,
        average="weighted",
    )

    cm = confusion_matrix(y_test, test_predictions)

    print("\n")
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print(f"Training Accuracy : {train_accuracy:.4f}")
    print(f"Testing Accuracy  : {test_accuracy:.4f}")
    print(f"Precision         : {precision:.4f}")
    print(f"Recall            : {recall:.4f}")
    print(f"F1 Score          : {f1:.4f}")

    print("\nConfusion Matrix")
    print("-" * 40)
    print(cm)

    print("\nClassification Report")
    print("-" * 40)
    print(classification_report(y_test, test_predictions))


def main():
    """
    Main Function
    """

    X, y, iris = load_dataset()

    explore_dataset(iris)

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    model = train_model(X_train, y_train)

    evaluate_model(
        model,
        X_train,
        X_test,
        y_train,
        y_test,
    )


if __name__ == "__main__":
    main()