# ==========================================================
# MLBench Summer Internship - Day 11
#
# Project:
# Iris Dataset Exploration
#
# Description:
# Load and explore the Iris dataset using Pandas.
#
# Author: Hadeed Jalani
# ==========================================================

from sklearn.datasets import load_iris
import pandas as pd


def load_dataset():
    """
    Load the Iris dataset.
    """

    iris = load_iris()

    print("=" * 70)
    print("Loading Iris Dataset...")
    print("Dataset loaded successfully!")
    print("=" * 70)

    return iris


def create_dataframe(iris):
    """
    Convert dataset into a Pandas DataFrame.
    """

    dataframe = pd.DataFrame(
        iris.data,
        columns=iris.feature_names,
    )

    dataframe["target"] = iris.target

    return dataframe


def explore_dataset(dataframe, iris):
    """
    Display dataset information.
    """

    print("\n")
    print("=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)

    print(f"Dataset Shape      : {dataframe.shape}")
    print(f"Total Samples      : {len(dataframe)}")
    print(f"Total Features     : {len(iris.feature_names)}")
    print(f"Target Classes     : {list(iris.target_names)}")

    print("\n")
    print("=" * 70)
    print("FIRST FIVE RECORDS")
    print("=" * 70)

    print(dataframe.head())

    print("\n")
    print("=" * 70)
    print("DATAFRAME INFORMATION")
    print("=" * 70)

    dataframe.info()

    print("\n")
    print("=" * 70)
    print("STATISTICAL SUMMARY")
    print("=" * 70)

    print(dataframe.describe())

    print("\n")
    print("=" * 70)
    print("TARGET CLASS DISTRIBUTION")
    print("=" * 70)

    print(dataframe["target"].value_counts())

    print("\nTarget Labels")

    for index, name in enumerate(iris.target_names):

        print(f"{index} -> {name}")


def main():

    iris = load_dataset()

    dataframe = create_dataframe(iris)

    explore_dataset(
        dataframe,
        iris,
    )


if __name__ == "__main__":

    main()