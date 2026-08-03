# ==================================================
#             JSON PRACTICE
#            MLBench Summer Internship
# ==================================================
import json
import os
FILE_NAME = "Day4/students.json"
def create_json():

    students = {
        "students": [
            {
                "id": 1,
                "name": "Ali",
                "age": 20,
                "marks": 87
            },
            {
                "id": 2,
                "name": "Ahmed",
                "age": 21,
                "marks": 92
            }
        ]
    }

    with open(FILE_NAME, "w") as file:
        json.dump(students, file, indent=4)

    print("\nStudent data saved successfully.")

def read_json():

    if not os.path.exists(FILE_NAME):
        print("\nJSON file does not exist.")
        return

    with open(FILE_NAME, "r") as file:
        students = json.load(file)

    print("\nStudent Records\n")

    for student in students["students"]:

        print("-" * 30)
        print(f"ID    : {student['id']}")
        print(f"Name  : {student['name']}")
        print(f"Age   : {student['age']}")
        print(f"Marks : {student['marks']}")

def add_student():

    if not os.path.exists(FILE_NAME):
        print("\nCreate the JSON file first.")
        return

    with open(FILE_NAME, "r") as file:
        students = json.load(file)

    try:
        student = {
            "id": int(input("Enter ID: ")),
            "name": input("Enter Name: "),
            "age": int(input("Enter Age: ")),
            "marks": float(input("Enter Marks: "))
        }

    except ValueError:
        print("\nInvalid input! Please enter numbers where required.")
        return

    students["students"].append(student)

    with open(FILE_NAME, "w") as file:
        json.dump(students, file, indent=4)

    print("\nStudent Added Successfully.")
    
def update_student():

    if not os.path.exists(FILE_NAME):
        print("\nJSON file not found.")
        return

    # Validate Student ID
    try:
        student_id = int(input("\nEnter Student ID to Update: "))
    except ValueError:
        print("\nInvalid ID! Please enter a numeric value.")
        return

    with open(FILE_NAME, "r") as file:
        students = json.load(file)

    found = False

    for student in students["students"]:

        if student["id"] == student_id:

            try:
                student["name"] = input("Enter New Name: ")
                student["age"] = int(input("Enter New Age: "))
                student["marks"] = float(input("Enter New Marks: "))

            except ValueError:
                print("\nInvalid input! Age must be an integer and Marks must be a number.")
                return

            found = True
            break

    if found:

        with open(FILE_NAME, "w") as file:
            json.dump(students, file, indent=4)

        print("\nStudent Updated Successfully.")

    else:

        print("\nStudent Not Found.")

def main():

    while True:

        print("\n" + "=" * 55)
        print("              JSON PRACTICE")
        print("=" * 55)

        print("1. Create Student JSON")
        print("2. Read Student JSON")
        print("3. Add Student")
        print("4. Update Student")
        print("5. Exit")

        choice = input("\nEnter your choice: ")

        if choice == "1":
            create_json()

        elif choice == "2":
            read_json()

        elif choice == "3":
            add_student()

        elif choice == "4":
            update_student()

        elif choice == "5":
            print("\nThank you!")
            break

        else:
            print("\nInvalid Choice!")

if __name__ == "__main__":
    main()
