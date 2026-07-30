# Function to calculate grade based on average marks
def calculate_grade(average):
    if average >= 85:
        return "A"
    elif average >= 80:
        return "A-"
    elif average >= 75:
        return "B+"
    elif average >= 70:
        return "B-"
    elif average >= 65:
        return "C+"
    elif average >= 60:
        return "C-"
    elif average >= 55:
        return "D+"
    elif average >= 50:
        return "D-"
    else:
        return "F"


# Display program title
print("=" * 50)
print("           STUDENT GRADING SYSTEM")
print("=" * 50)

# Collect student information
student_name = input("Enter Student Name: ")
student_class = input("Enter Class: ")
number_of_subjects = int(input("Enter Number of Subjects: "))

# Dictionary to store subjects and marks
subjects = {}

# Store subject names and marks
for i in range(number_of_subjects):
    subject_name = input(f"\nEnter Subject {i + 1} Name: ")

    while True:
        subject_marks = float(input(f"Enter Marks in {subject_name}: "))

        if 0 <= subject_marks <= 100:
            break

        print("Invalid input! Marks must be between 0 and 100.")

    # Store marks after validation
    subjects[subject_name] = subject_marks

# Calculate total and average marks
total_marks = sum(subjects.values())
average_marks = total_marks / number_of_subjects

# Calculate grade
grade = calculate_grade(average_marks)

# Determine pass/fail status
status = "PASS" if grade != "F" else "FAIL"

# Display report
print("\n" + "=" * 50)
print("               STUDENT REPORT")
print("=" * 50)

print(f"Student Name : {student_name}")
print(f"Class        : {student_class}")

print("\nSubjects and Marks")
print("-" * 50)

for subject, marks in subjects.items():
    print(f"{subject:<20} : {marks}")

print("-" * 50)
print(f"Total Marks     : {total_marks}")
print(f"Average Marks   : {average_marks:.2f}")
print(f"Final Grade     : {grade}")
print(f"Status          : {status}")

print("=" * 50)