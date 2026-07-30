# ==========================================================
#                  MLBench Summer Internship
#                          Day 9
# ==========================================================
#
# Project:
# Iris Flower Classification System
#
# Description:
# A Machine Learning classification system that predicts
# the species of an Iris flower using Logistic Regression.
#
# Dataset:
# Iris Dataset
# Source:
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
#
# Author:
# Hadeed Jalani
#
# ==========================================================


# ==========================================================
# Imports
# ==========================================================

import joblib

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


# ==========================================================
# Utility Functions
# ==========================================================

def print_header(title):
    """
    Display a formatted section header.
    """

    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_subtitle(title):
    """
    Display a formatted subsection title.
    """

    print("\n" + title)
    print("-" * 40)


# ==========================================================
# Dataset Functions
# ==========================================================

def load_dataset():
    """
    Load the Iris dataset from Scikit-Learn.
    """

    print_header("LOADING IRIS DATASET")

    iris = load_iris()

    print("Dataset loaded successfully.")
    print(f"Dataset Shape : {iris.data.shape}")

    return iris


def explore_dataset(iris):
    """
    Display dataset information.
    """

    print_header("IRIS DATASET INFORMATION")

    print(f"Total Samples  : {len(iris.data)}")
    print(f"Total Features : {len(iris.feature_names)}")
    print(f"Total Classes  : {len(iris.target_names)}")

    print_subtitle("Feature Names")

    for feature in iris.feature_names:
        print(f"• {feature}")

    print_subtitle("Target Classes")

    for target in iris.target_names:
        print(f"• {target}")

    print_subtitle("First Five Records")

    for index in range(5):

        print(
            f"Features : {iris.data[index]}"
            f" | Species : {iris.target_names[iris.target[index]]}"
        )


# ==========================================================
# Dataset Splitting
# ==========================================================

def split_dataset(X, y):
    """
    Split the dataset into training and testing sets.
    """

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y,

    )

    print_header("DATA SPLITTING")

    print(f"Training Samples : {len(X_train)}")
    print(f"Testing Samples  : {len(X_test)}")

    return (

        X_train,
        X_test,
        y_train,
        y_test,

    )
    # ==========================================================
# Model Training Functions
# ==========================================================

def train_model(model, X_train, y_train):
    """
    Generic function to train any classification model.
    """

    model.fit(
        X_train,
        y_train,
    )

    return model


def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression model.
    """

    print_header("TRAINING LOGISTIC REGRESSION MODEL")

    model = LogisticRegression(
        max_iter=200,
        random_state=42,
    )

    trained_model = train_model(
        model,
        X_train,
        y_train,
    )

    joblib.dump(
        trained_model,
        "logistic_regression_model.pkl",
    )

    print("✓ Logistic Regression model trained successfully.")
    print("✓ Model saved as logistic_regression_model.pkl")

    return trained_model


def train_decision_tree(X_train, y_train):
    """
    Train a Decision Tree classifier.
    """

    print_header("TRAINING DECISION TREE MODEL")

    model = DecisionTreeClassifier(
        random_state=42,
    )

    trained_model = train_model(
        model,
        X_train,
        y_train,
    )

    print("✓ Decision Tree model trained successfully.")

    return trained_model


# ==========================================================
# Model Evaluation
# ==========================================================

def evaluate_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    model_name,
):
    """
    Evaluate a trained classification model.
    """

    train_predictions = model.predict(
        X_train
    )

    test_predictions = model.predict(
        X_test
    )

    train_accuracy = accuracy_score(
        y_train,
        train_predictions,
    )

    test_accuracy = accuracy_score(
        y_test,
        test_predictions,
    )

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

    confusion = confusion_matrix(
        y_test,
        test_predictions,
    )

    print_header(
        f"{model_name.upper()} MODEL EVALUATION"
    )

    print(f"Training Accuracy : {train_accuracy:.4f}")
    print(f"Testing Accuracy  : {test_accuracy:.4f}")
    print(f"Precision         : {precision:.4f}")
    print(f"Recall            : {recall:.4f}")
    print(f"F1 Score          : {f1:.4f}")

    print_subtitle(
        "Classification Report"
    )

    print(
        classification_report(
            y_test,
            test_predictions,
            target_names=[
                "Setosa",
                "Versicolor",
                "Virginica",
            ],
        )
    )

    return {

        "accuracy": test_accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "train_accuracy": train_accuracy,

        "confusion_matrix": confusion,

        "predictions": test_predictions,

    }


# ==========================================================
# Confusion Matrix
# ==========================================================

def save_confusion_matrix(
    confusion,
    class_names,
):
    """
    Display and save the confusion matrix.
    """

    print_header(
        "GENERATING CONFUSION MATRIX"
    )

    display = ConfusionMatrixDisplay(

        confusion_matrix=confusion,

        display_labels=class_names,

    )

    display.plot(
        cmap="Blues",
        values_format="d",
    )

    plt.title(
        "Logistic Regression Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        "logistic_regression_confusion_matrix.png",
        dpi=300,
    )

    plt.show()

    plt.close()

    print(
        "✓ Confusion Matrix saved successfully."
    )

    print(
        "✓ File: logistic_regression_confusion_matrix.png"
    )

# ==========================================================
# Model Comparison
# ==========================================================

def compare_models(
    logistic_results,
    decision_tree_results,
):
    """
    Compare the performance of the trained models.
    """

    print_header("MODEL COMPARISON")

    print(
        f"{'Metric':<20}"
        f"{'Logistic Regression':<25}"
        f"{'Decision Tree'}"
    )

    print("-" * 70)

    metrics = [

        ("Training Accuracy", "train_accuracy"),

        ("Testing Accuracy", "accuracy"),

        ("Precision", "precision"),

        ("Recall", "recall"),

        ("F1 Score", "f1"),

    ]

    for metric_name, key in metrics:

        print(

            f"{metric_name:<20}"

            f"{logistic_results[key]:<25.4f}"

            f"{decision_tree_results[key]:.4f}"

        )


# ==========================================================
# Sample Predictions
# ==========================================================

def display_sample_predictions(
    model,
    X_test,
    y_test,
    iris,
):
    """
    Display sample predictions.
    """

    predictions = model.predict(
        X_test
    )

    print_header(
        "SAMPLE PREDICTIONS"
    )

    feature_names = iris.feature_names

    samples_to_display = min(10, len(X_test))

    for i in range(samples_to_display):

        print(f"\nSample {i + 1}")

        print("-" * 40)

        for feature, value in zip(
            feature_names,
            X_test[i],
        ):

            print(
                f"{feature:<20}: {value:.1f}"
            )

        actual = iris.target_names[
            y_test[i]
        ]

        predicted = iris.target_names[
            predictions[i]
        ]

        print()

        print(
            f"Actual Species    : {actual}"
        )

        print(
            f"Predicted Species : {predicted}"
        )

        if actual == predicted:

            print(
                "Prediction Status : ✓ Correct"
            )

        else:

            print(
                "Prediction Status : ✗ Incorrect"
            )


# ==========================================================
# Project Summary
# ==========================================================

def display_summary(
    logistic_results,
    decision_tree_results,
):
    """
    Display final project summary.
    """

    print_header(
        "PROJECT SUMMARY"
    )

    if (
        logistic_results["accuracy"]
        >
        decision_tree_results["accuracy"]
    ):

        best_model = "Logistic Regression"

    elif (
        logistic_results["accuracy"]
        <
        decision_tree_results["accuracy"]
    ):

        best_model = "Decision Tree"

    else:

        best_model = "Both Models"

    print(
        f"Best Performing Model : {best_model}"
    )

    print()

    print("Final Results")

    print("-" * 30)

    print(
        f"Logistic Regression Accuracy : "
        f"{logistic_results['accuracy']:.4f}"
    )

    print(
        f"Decision Tree Accuracy       : "
        f"{decision_tree_results['accuracy']:.4f}"
    )

    print()

    print("Observations")

    print("-" * 30)

    if (
        decision_tree_results["train_accuracy"]
        >
        decision_tree_results["accuracy"]
    ):

        print(
            "• Decision Tree shows signs of overfitting."
        )

    else:

        print(
            "• Decision Tree generalized well."
        )

    if (
        abs(
            logistic_results["train_accuracy"]
            -
            logistic_results["accuracy"]
        )
        < 0.05
    ):

        print(
            "• Logistic Regression generalized well."
        )

    print(
        "• Logistic Regression is recommended for the Iris Dataset."
    )

    print(
        "• The model achieved excellent classification performance."
    )


# ==========================================================
# Main Function
# ==========================================================

def main():

    iris = load_dataset()

    explore_dataset(
        iris
    )

    X = iris.data

    y = iris.target

    X_train, X_test, y_train, y_test = split_dataset(

        X,

        y,

    )

    logistic_model = train_logistic_regression(

        X_train,

        y_train,

    )

    decision_tree_model = train_decision_tree(

        X_train,

        y_train,

    )

    logistic_results = evaluate_model(

        logistic_model,

        X_train,

        X_test,

        y_train,

        y_test,

        "Logistic Regression",

    )

    decision_tree_results = evaluate_model(

        decision_tree_model,

        X_train,

        X_test,

        y_train,

        y_test,

        "Decision Tree",

    )

    save_confusion_matrix(

        logistic_results["confusion_matrix"],

        iris.target_names,

    )

    compare_models(

        logistic_results,

        decision_tree_results,

    )

    display_sample_predictions(

        logistic_model,

        X_test,

        y_test,

        iris,

    )

    display_summary(

        logistic_results,

        decision_tree_results,

    )

    print_header(
        "PROJECT COMPLETED SUCCESSFULLY"
    )

    print(
        "✓ Iris Dataset Loaded"
    )

    print(
        "✓ Models Trained Successfully"
    )

    print(
        "✓ Models Evaluated"
    )

    print(
        "✓ Confusion Matrix Generated"
    )

    print(
        "✓ Logistic Regression Model Saved (.pkl)"
    )

    print(
        "✓ Project Ready for Streamlit Deployment"
    )

    print(
        "✓ GitHub Repository Ready"
    )


# ==========================================================
# Program Entry Point
# ==========================================================

if __name__ == "__main__":

    main()    