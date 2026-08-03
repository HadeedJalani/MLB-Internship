# ==========================================================
# MLBench Summer Internship - Day 9
#
# Train Logistic Regression Model
#
# This script trains a Logistic Regression model
# on the Iris Dataset and saves it as a .pkl file
# for use in the Streamlit application.
#
# Author: Hadeed Jalani
# ==========================================================

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib


from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib

iris = load_iris()

X = iris.data
y = iris.target

model = LogisticRegression(
    max_iter=200,
    random_state=42,
)

model.fit(X, y)

joblib.dump(
    model,
    "logistic_regression_model.pkl",
)

print("Model saved successfully!")

if __name__ == "__main__":
    main()