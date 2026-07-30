# ==================================================
#      STUDENT RECORD MANAGEMENT SYSTEM
#          MLBench Summer Internship
# ==================================================

import json
import os

FILE_NAME = "Day4/students.json"

def load_students():

    if not os.path.exists(FILE_NAME):
        return []

    with open(FILE_NAME, "r") as file:
        data = json.load(file)

    return data["students"]

def save_students(students):

    with open(FILE_NAME, "w") as file:
        json.dump({"students": students}, file, indent=4)

def view_students(students):

    if not students:
        print("\nNo student records found.")
        return

    print("\n" + "=" * 60)
    print("               STUDENT RECORDS")
    print("=" * 60)

    for student in students:

        print(f"\nID     : {student['id']}")
        print(f"Name   : {student['name']}")
        print(f"Age    : {student['age']}")
        print(f"Marks  : {student['marks']}")

    print("=" * 60)

def add_student(students):

    try:

        student = {
            "id": int(input("\nEnter ID: ")),
            "name": input("Enter Name: "),
            "age": int(input("Enter Age: ")),
            "marks": float(input("Enter Marks: "))
        }

    except ValueError:

        print("\nInvalid Input!")
        return

    for s in students:

        if s["id"] == student["id"]:
            print("\nStudent ID already exists.")
            return

    students.append(student)

    save_students(students)

    print("\nStudent Added Successfully.")

def search_student(students):

    try:
        student_id = int(input("\nEnter Student ID to Search: "))
    except ValueError:
        print("\nInvalid ID!")
        return

    for student in students:
        if student["id"] == student_id:

            print("\nStudent Found")
            print("-" * 40)
            print(f"ID     : {student['id']}")
            print(f"Name   : {student['name']}")
            print(f"Age    : {student['age']}")
            print(f"Marks  : {student['marks']}")
            return

    print("\nStudent Not Found.")

def update_student(students):

    try:
        student_id = int(input("\nEnter Student ID to Update: "))
    except ValueError:
        print("\nInvalid ID!")
        return

    for student in students:

        if student["id"] == student_id:

            try:
                student["name"] = input("Enter New Name: ")
                student["age"] = int(input("Enter New Age: "))
                student["marks"] = float(input("Enter New Marks: "))
            except ValueError:
                print("\nInvalid Input!")
                return

            save_students(students)
            print("\nStudent Updated Successfully.")
            return

    print("\nStudent Not Found.")

def delete_student(students):

    try:
        student_id = int(input("\nEnter Student ID to Delete: "))
    except ValueError:
        print("\nInvalid ID!")
        return

    for student in students:

        if student["id"] == student_id:
            students.remove(student)
            save_students(students)
            print("\nStudent Deleted Successfully.")
            return

    print("\nStudent Not Found.")

def main():

    students = load_students()

    while True:

        print("\n" + "=" * 55)
        print("     STUDENT RECORD MANAGEMENT SYSTEM")
        print("=" * 55)

        print("1. View All Students")
        print("2. Add Student")
        print("3. Search Student")
        print("4. Update Student")
        print("5. Delete Student")
        print("6. Exit")

        choice = input("\nEnter your choice: ")

        if choice == "1":
            view_students(students)

        elif choice == "2":
            add_student(students)

        elif choice == "3":
            search_student(students)

        elif choice == "4":
            update_student(students)

        elif choice == "5":
            delete_student(students)

        elif choice == "6":
            print("\nThank you!")
            break

        else:
            print("\nInvalid Choice! Please try again.")                

if __name__ == "__main__":
    main()        