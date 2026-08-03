# ==================================================
#           PANDAS PRACTICE - DAY 6
#          MLBench Summer Internship
# ==================================================

import pandas as pd

FILE_NAME = "Day6/student_performance.csv"


def load_dataset():

    print("\n===== Loading Dataset =====")

    data = pd.read_csv(FILE_NAME)

    print("\nDataset Loaded Successfully!")

    return data


def display_rows(data):

    print("\n===== First Five Rows =====")
    print(data.head())

    print("\n===== Last Five Rows =====")
    print(data.tail())


def dataset_information(data):

    print("\n===== Dataset Information =====")

    data.info()

    print("\n===== Summary Statistics =====")

    print(data.describe())


def missing_values(data):

    print("\n===== Missing Values =====")

    print(data.isnull().sum())


def filter_data(data):

    print("\n===== Students with Python Marks >= 80 =====")

    filtered = data[data["Python"] >= 80]

    print(filtered)


def select_columns(data):

    print("\n===== Name and Python Marks =====")

    print(data[["Name", "Python"]])


def main():

    data = load_dataset()

    display_rows(data)

    dataset_information(data)

    missing_values(data)

    filter_data(data)

    select_columns(data)


if __name__ == "__main__":
    main()