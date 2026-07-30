# ==================================================
#          DATA VISUALIZATION USING PANDAS
#            MLBench Summer Internship
# ==================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if not os.path.exists("charts"):
    os.makedirs("charts")

data = pd.read_csv("cleaned_student_performance.csv")

print("Dataset Loaded Successfully!")

plt.figure(figsize=(10,6))

plt.bar(data["Name"], data["Average_Score"])

plt.title("Average Score Per Student")
plt.xlabel("Student")
plt.ylabel("Average Score")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("charts/bar_chart.png")

plt.show()

plt.figure(figsize=(8,5))

plt.hist(data["Average_Score"], bins=10)

plt.title("Distribution of Average Scores")
plt.xlabel("Average Score")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig("charts/histogram.png")

plt.show()

plt.figure(figsize=(8,6))

plt.scatter(data["Python"], data["ML"])

plt.title("Python vs Machine Learning")
plt.xlabel("Python Marks")
plt.ylabel("Machine Learning Marks")

plt.tight_layout()

plt.savefig("charts/scatter_plot.png")

plt.show()

performance = data["Performance"].value_counts()

plt.figure(figsize=(7,7))

plt.pie(
    performance,
    labels=performance.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Performance Categories")

plt.savefig("charts/pie_chart.png")

plt.show()

plt.figure(figsize=(8,6))

sns.boxplot(
    data=data[["Python","Mathematics","Statistics","ML"]]
)

plt.title("Marks Distribution Across Subjects")

plt.tight_layout()

plt.savefig("charts/box_plot.png")

plt.show()

