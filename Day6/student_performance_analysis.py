# ==================================================
#      STUDENT PERFORMANCE ANALYSIS - DAY 6
#          MLBench Summer Internship
# ==================================================

import pandas as pd

FILE_NAME = "Day6/student_performance.csv"
OUTPUT_FILE = "Day6/processed_student_performance.csv"


def load_dataset():

    data = pd.read_csv(FILE_NAME)

    print("\nDataset Loaded Successfully!")

    return data


def dataset_information(data):

    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)

    data.info()

    print("\nTotal Students:", len(data))

def average_marks(data):

    print("\n" + "=" * 60)
    print("AVERAGE MARKS")
    print("=" * 60)

    subjects = [
        "Python",
        "Mathematics",
        "Statistics",
        "Machine_Learning"
    ]

    for subject in subjects:

        print(f"{subject}: {data[subject].mean():.2f}")

def top_students(data):

    print("\n" + "=" * 60)
    print("TOP 5 STUDENTS")
    print("=" * 60)

    data["Average"] = data[
        [
            "Python",
            "Mathematics",
            "Statistics",
            "Machine_Learning"
        ]
    ].mean(axis=1)

    top = data.sort_values(by="Average", ascending=False).head(5)

    print(top[["Student_ID", "Name", "Average"]])

    return data

def below_average_students(data):

    overall_average = data["Average"].mean()

    print("\n" + "=" * 60)
    print("STUDENTS BELOW CLASS AVERAGE")
    print("=" * 60)

    below = data[data["Average"] < overall_average]

    print(below[["Student_ID", "Name", "Average"]])

def save_dataset(data):

    data.to_csv(OUTPUT_FILE, index=False)

    print(f"\nProcessed dataset saved as '{OUTPUT_FILE}'")

def main():

    data = load_dataset()

    dataset_information(data)

    average_marks(data)

    data = top_students(data)

    below_average_students(data)

    save_dataset(data)


if __name__ == "__main__":
    main()

    
