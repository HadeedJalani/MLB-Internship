# ==================================================
#           STUDENT ANALYTICS BACKEND
# ==================================================

import pandas as pd
import matplotlib.pyplot as plt
import os

DATASET = "student_performance.csv"
CLEANED_DATASET = "cleaned_student_performance.csv"


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

def load_dataset():

    data = pd.read_csv(DATASET)

    return data


# --------------------------------------------------
# Clean Dataset
# --------------------------------------------------

def clean_dataset():

    data = pd.read_csv(DATASET)

    # Remove duplicates

    data = data.drop_duplicates()

    # Fill missing numeric values

    numeric_cols = [
        "Python",
        "Mathematics",
        "Statistics",
        "Machine_Learning",
        "Attendance"
    ]

    for col in numeric_cols:

        data[col] = data[col].fillna(data[col].mean())

    # Fill missing strings

    data["Program"] = data["Program"].fillna("Unknown")

    data["Name"] = data["Name"].fillna("Unknown")

    # Average Score

    data["Average_Score"] = (

        data["Python"] +
        data["Mathematics"] +
        data["Statistics"] +
        data["Machine_Learning"]

    ) / 4

    # Performance Category

    performance = []

    for score in data["Average_Score"]:

        if score >= 90:
            performance.append("Excellent")

        elif score >= 80:
            performance.append("Good")

        elif score >= 70:
            performance.append("Average")

        else:
            performance.append("Needs Improvement")

    data["Performance"] = performance

    data.to_csv(CLEANED_DATASET, index=False)

    return data

# --------------------------------------------------
# Dashboard Statistics
# --------------------------------------------------

def generate_dashboard():

    data = clean_dataset()

    total_students = len(data)

    avg_python = round(data["Python"].mean(), 2)
    avg_math = round(data["Mathematics"].mean(), 2)
    avg_stats = round(data["Statistics"].mean(), 2)
    avg_ml = round(data["Machine_Learning"].mean(), 2)
    avg_attendance = round(data["Attendance"].mean(), 2)

    top_student = data.loc[data["Average_Score"].idxmax()]

    lowest_student = data.loc[data["Average_Score"].idxmin()]

    highest_subject = {

        "Python": avg_python,
        "Mathematics": avg_math,
        "Statistics": avg_stats,
        "Machine Learning": avg_ml

    }

    highest_average_subject = max(
        highest_subject,
        key=highest_subject.get
    )

    top5 = data.nlargest(
        5,
        "Average_Score"
    )[["Name", "Average_Score"]]

    bottom5 = data.nsmallest(
        5,
        "Average_Score"
    )[["Name", "Average_Score"]]

    performance = (
        data["Performance"]
        .value_counts()
        .to_string()
    )

    program_distribution = (
        data["Program"]
        .value_counts()
        .to_string()
    )

    report = f"""

==================================================
STUDENT PERFORMANCE ANALYTICS
==================================================

Total Students : {total_students}

Average Python Marks : {avg_python}

Average Mathematics Marks : {avg_math}

Average Statistics Marks : {avg_stats}

Average Machine Learning Marks : {avg_ml}

Average Attendance : {avg_attendance}%

Highest Average Subject :
{highest_average_subject}

Top Performer :
{top_student['Name']} ({top_student['Average_Score']:.2f})

Lowest Performer :
{lowest_student['Name']} ({lowest_student['Average_Score']:.2f})

==================================================
TOP 5 STUDENTS
==================================================

{top5.to_string(index=False)}

==================================================
BOTTOM 5 STUDENTS
==================================================

{bottom5.to_string(index=False)}

==================================================
PROGRAM DISTRIBUTION
==================================================

{program_distribution}

==================================================
PERFORMANCE DISTRIBUTION
==================================================

{performance}

"""

    return report    

# --------------------------------------------------
# Generate Charts
# --------------------------------------------------

def generate_charts():

    data = clean_dataset()

    os.makedirs("charts", exist_ok=True)

    # -------------------------------
    # Bar Chart
    # -------------------------------

    plt.figure(figsize=(12,5))

    plt.bar(data["Name"], data["Average_Score"])

    plt.xticks(rotation=90)

    plt.title("Average Score Per Student")

    plt.tight_layout()

    plt.savefig("charts/bar_chart.png")

    plt.close()


    # -------------------------------
    # Histogram
    # -------------------------------

    plt.figure(figsize=(8,5))

    plt.hist(data["Average_Score"], bins=8)

    plt.title("Average Score Distribution")

    plt.tight_layout()

    plt.savefig("charts/histogram.png")

    plt.close()


    # -------------------------------
    # Scatter Plot
    # -------------------------------

    plt.figure(figsize=(8,5))

    plt.scatter(
        data["Python"],
        data["Machine_Learning"]
    )

    plt.xlabel("Python")

    plt.ylabel("Machine Learning")

    plt.title("Python vs Machine Learning")

    plt.tight_layout()

    plt.savefig("charts/scatter_plot.png")

    plt.close()


    # -------------------------------
    # Pie Chart
    # -------------------------------

    plt.figure(figsize=(6,6))

    data["Performance"].value_counts().plot(
        kind="pie",
        autopct="%1.1f%%"
    )

    plt.ylabel("")

    plt.title("Performance Distribution")

    plt.tight_layout()

    plt.savefig("charts/pie_chart.png")

    plt.close()


    # -------------------------------
    # Box Plot
    # -------------------------------

    plt.figure(figsize=(8,5))

    plt.boxplot([
        data["Python"],
        data["Mathematics"],
        data["Statistics"],
        data["Machine_Learning"]
    ])

    plt.xticks(
        [1,2,3,4],
        [
            "Python",
            "Math",
            "Statistics",
            "ML"
        ]
    )

    plt.title("Subject Marks Distribution")

    plt.tight_layout()

    plt.savefig("charts/box_plot.png")

    plt.close()


    return [
        "charts/bar_chart.png",
        "charts/histogram.png",
        "charts/scatter_plot.png",
        "charts/pie_chart.png",
        "charts/box_plot.png"
    ]    