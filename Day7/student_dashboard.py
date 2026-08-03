# ==================================================
#        STUDENT PERFORMANCE DASHBOARD
#            MLBench Summer Internship
# ==================================================

import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
data = pd.read_csv("cleaned_student_performance.csv")

print("=" * 60)
print("STUDENT PERFORMANCE DASHBOARD")
print("=" * 60)

# ---------------------------------------
# Total Students
# ---------------------------------------

print(f"\nTotal Students: {len(data)}")

# ---------------------------------------
# Average Score Per Subject
# ---------------------------------------

print("\nAverage Score Per Subject")

subjects = [
    "Python",
    "Mathematics",
    "Statistics",
    "ML"
]

for subject in subjects:
    print(f"{subject}: {data[subject].mean():.2f}")

# ---------------------------------------
# Top 5 Students
# ---------------------------------------

print("\nTop 5 Students")

top_students = data.nlargest(5, "Average_Score")

print(
    top_students[
        ["Name", "Average_Score"]
    ]
)

# ---------------------------------------
# Students Needing Improvement
# ---------------------------------------

print("\nStudents Needing Improvement")

needs_improvement = data[
    data["Performance"] == "Needs Improvement"
]

print(
    needs_improvement[
        ["Name", "Average_Score"]
    ]
)

# ---------------------------------------
# Subject with Highest Average
# ---------------------------------------

subject_averages = {
    "Python": data["Python"].mean(),
    "Mathematics": data["Mathematics"].mean(),
    "Statistics": data["Statistics"].mean(),
    "ML": data["ML"].mean()
}

highest_subject = max(
    subject_averages,
    key=subject_averages.get
)

print(f"\nHighest Average Subject: {highest_subject}")

# ---------------------------------------
# Visualization
# ---------------------------------------

plt.figure(figsize=(8,5))

plt.bar(
    subject_averages.keys(),
    subject_averages.values()
)

plt.title("Average Marks Per Subject")

plt.xlabel("Subjects")

plt.ylabel("Average Marks")

plt.tight_layout()

plt.show()