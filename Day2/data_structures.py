def list_operations():
    print("\n" + "=" * 55)
    print("               LIST OPERATIONS")
    print("=" * 55)

    numbers = list(map(int, input("Enter numbers separated by spaces: ").split()))

    while True:
        print("\nChoose an Operation:")
        print("1. Find Largest Number")
        print("2. Find Second Largest Number")
        print("3. Remove Duplicates")
        print("4. Reverse List")
        print("5. Find Common Elements")
        print("6. Back to Main Menu")

        choice = input("\nEnter your choice: ")

        if choice == "1":
            print(f"\nLargest Number: {max(numbers)}")

        elif choice == "2":
            unique_numbers = list(set(numbers))

            if len(unique_numbers) < 2:
                print("\nSecond largest number does not exist.")
            else:
                unique_numbers.sort(reverse=True)
                print(f"\nSecond Largest Number: {unique_numbers[1]}")

        elif choice == "3":
            unique_list = []

            for num in numbers:
                if num not in unique_list:
                    unique_list.append(num)

            print(f"\nList after removing duplicates:\n{unique_list}")

        elif choice == "4":
            reversed_list = []

            for i in range(len(numbers) - 1, -1, -1):
                reversed_list.append(numbers[i])

            print(f"\nReversed List:\n{reversed_list}")

        elif choice == "5":
            second_list = list(map(int, input("\nEnter second list: ").split()))

            common = []

            for num in numbers:
                if num in second_list and num not in common:
                    common.append(num)

            print(f"\nCommon Elements:\n{common}")

        elif choice == "6":
            break

        else:
            print("Invalid choice. Please try again.")

def tuple_operations():
    print("\n" + "=" * 55)
    print("              TUPLE OPERATIONS")
    print("=" * 55)

    numbers = tuple(map(int, input("Enter tuple elements separated by spaces: ").split()))

    while True:
        print("\nChoose an Operation:")
        print("1. Count Occurrences of an Element")
        print("2. Convert Tuple to List")
        print("3. Convert List to Tuple")
        print("4. Back to Main Menu")

        choice = input("\nEnter your choice: ")

        if choice == "1":
            element = int(input("Enter element to count: "))
            count = numbers.count(element)
            print(f"\n{element} occurs {count} time(s).")

        elif choice == "2":
            converted_list = list(numbers)
            print("\nTuple converted to List:")
            print(converted_list)

        elif choice == "3":
            values = list(map(int, input("Enter list elements: ").split()))
            converted_tuple = tuple(values)

            print("\nList converted to Tuple:")
            print(converted_tuple)

        elif choice == "4":
            break

        else:
            print("Invalid choice. Please try again.")

def set_operations():
    print("\n" + "=" * 55)
    print("               SET OPERATIONS")
    print("=" * 55)

    numbers = list(map(int, input("Enter list elements separated by spaces: ").split()))
    number_set = set(numbers)

    while True:
        print("\nChoose an Operation:")
        print("1. Find Unique Values")
        print("2. Union of Two Sets")
        print("3. Intersection of Two Sets")
        print("4. Difference of Two Sets")
        print("5. Back to Main Menu")

        choice = input("\nEnter your choice: ")

        if choice == "1":
            print("\nUnique Values:")
            print(number_set)

        elif choice == "2":
            second_set = set(map(int, input("Enter second set: ").split()))
            print("\nUnion:")
            print(number_set.union(second_set))

        elif choice == "3":
            second_set = set(map(int, input("Enter second set: ").split()))
            print("\nIntersection:")
            print(number_set.intersection(second_set))

        elif choice == "4":
            second_set = set(map(int, input("Enter second set: ").split()))
            print("\nDifference (First - Second):")
            print(number_set.difference(second_set))

        elif choice == "5":
            break

        else:
            print("Invalid choice. Please try again.")

def dictionary_operations():
    print("\n" + "=" * 55)
    print("            DICTIONARY OPERATIONS")
    print("=" * 55)

    while True:
        print("\nChoose an Operation:")
        print("1. Create Student Record")
        print("2. Calculate Average Marks")
        print("3. Word Frequency Counter")
        print("4. Back to Main Menu")

        choice = input("\nEnter your choice: ")

        if choice == "1":

            student = {}

            student["Name"] = input("Enter Student Name: ")
            student["Roll No"] = input("Enter Roll Number: ")
            student["Class"] = input("Enter Class: ")

            number_of_subjects = int(input("Enter Number of Subjects: "))

            subjects = {}

            for i in range(number_of_subjects):
                subject = input(f"Enter Subject {i + 1}: ")
                marks = float(input(f"Enter Marks in {subject}: "))
                subjects[subject] = marks

            student["Subjects"] = subjects

            print("\nStudent Record")
            print("-" * 40)

            for key, value in student.items():
                print(f"{key}: {value}")

        elif choice == "2":

            number = int(input("Enter Number of Students: "))

            students = {}

            for i in range(number):
                name = input(f"\nEnter Student {i + 1} Name: ")
                marks = float(input(f"Enter Marks of {name}: "))

                students[name] = marks

            average = sum(students.values()) / len(students)

            print("\nStudent Marks")

            for name, marks in students.items():
                print(f"{name} : {marks}")

            print(f"\nAverage Marks : {average:.2f}")

        elif choice == "3":

            sentence = input("\nEnter a sentence:\n").lower()

            words = sentence.split()

            frequency = {}

            for word in words:

                if word in frequency:
                    frequency[word] += 1
                else:
                    frequency[word] = 1

            print("\nWord Frequency")

            for word, count in frequency.items():
                print(f"{word} : {count}")

        elif choice == "4":
            break

        else:
            print("Invalid choice. Please try again.")

def main():
    while True:
        print("\n" + "=" * 55)
        print("      PYTHON DATA STRUCTURES TOOLKIT")
        print("=" * 55)

        print("1. List Operations")
        print("2. Tuple Operations")
        print("3. Set Operations")
        print("4. Dictionary Operations")
        print("5. Exit")

        choice = input("\nEnter your choice: ")

        if choice == "1":
            list_operations()

        elif choice == "2":
             tuple_operations()
        elif choice == "3":
            set_operations()
        elif choice == "4":
            dictionary_operations()

        elif choice == "5":
            print("\nThank you for using the toolkit.")
            break

        else:
            print("\nInvalid Choice!")

if __name__ == "__main__":
    main()