# ==================================================
#          DATA CLEANING USING PANDAS
#            MLBench Summer Internship
# ==================================================

import pandas as pd
# Load Dataset

data = pd.read_csv("student_performance.csv")

print("\nDataset Loaded Successfully!\n")

print("=" * 60)
print("FIRST FIVE ROWS")
print("=" * 60)

print(data.head())

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(data.info())

print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

print(data.describe())

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(data.isnull().sum())

print("\nRemoving Duplicate Records...")

data.drop_duplicates(inplace=True)

print("Duplicates Removed Successfully!")

data.rename(
    columns={
        "Machine_Learning": "ML"
    },
    inplace=True
)

print("\nColumns Renamed Successfully!")

data["Average_Score"] = (
    data["Python"] +
    data["Mathematics"] +
    data["Statistics"] +
    data["ML"]
) / 4

print("\nAverage Score Column Added!")

def performance(score):

    if score >= 90:
        return "Excellent"

    elif score >= 80:
        return "Good"

    elif score >= 70:
        return "Average"

    else:
        return "Needs Improvement"


data["Performance"] = data["Average_Score"].apply(performance)

print("Performance Column Added!")

data.sort_values(
    by="Average_Score",
    ascending=False,
    inplace=True
)

print("\nDataset Sorted Successfully!")

data.to_csv(
    "cleaned_student_performance.csv",
    index=False
)

print("\nCleaned Dataset Saved Successfully!")
