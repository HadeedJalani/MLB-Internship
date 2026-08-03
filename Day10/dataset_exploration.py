# ==========================================================
# MLBench Summer Internship - Day 10
#
# Dataset Exploration
#
# Breast Cancer Wisconsin Diagnostic Dataset
# ==========================================================

import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_dataset():
    """
    Load the Breast Cancer dataset.
    """

    dataset = load_breast_cancer()

    print("=" * 70)
    print("Loading Breast Cancer Dataset...")
    print("Dataset loaded successfully!")
    print("=" * 70)

    return dataset


def create_dataframe(dataset):
    """
    Convert dataset into a Pandas DataFrame.
    """

    dataframe = pd.DataFrame(
        dataset.data,
        columns=dataset.feature_names,
    )

    dataframe["target"] = dataset.target

    return dataframe


def explore_dataset(dataframe, dataset):
    """
    Display dataset information.
    """

    print("\n")
    print("=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)

    print(f"Shape              : {dataframe.shape}")
    print(f"Features           : {len(dataset.feature_names)}")
    print(f"Samples            : {len(dataframe)}")
    print(f"Target Classes     : {dataset.target_names}")

    print("\n")
    print("=" * 70)
    print("FIRST FIVE ROWS")
    print("=" * 70)

    print(dataframe.head())

    print("\n")
    print("=" * 70)
    print("DATAFRAME INFO")
    print("=" * 70)

    dataframe.info()

    print("\n")
    print("=" * 70)
    print("STATISTICAL SUMMARY")
    print("=" * 70)

    print(dataframe.describe())

    print("\n")
    print("=" * 70)
    print("TARGET DISTRIBUTION")
    print("=" * 70)

    print(dataframe["target"].value_counts())

    print("\nTarget Labels")

    for index, name in enumerate(dataset.target_names):
        print(f"{index} -> {name}")


def main():

    dataset = load_dataset()

    dataframe = create_dataframe(dataset)

    explore_dataset(
        dataframe,
        dataset,
    )


if __name__ == "__main__":
    main()