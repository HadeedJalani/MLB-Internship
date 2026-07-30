# ==================================================
#          FILE HANDLING PRACTICE
#            MLBench Summer Internship
# ==================================================
import os
FILE_NAME = "Day4/sample.txt"
def write_file():

    print("\nWrite Data to File")

    data = input("Enter text: ")

    with open(FILE_NAME, "w") as file:
        file.write(data + "\n")

    print("\nData written successfully.")

def read_file():

    if not os.path.exists(FILE_NAME):
        print("\nFile does not exist.")
        return

    print("\nFile Contents:\n")

    with open(FILE_NAME, "r") as file:
        print(file.read())

def append_file():

    data = input("\nEnter text to append: ")

    with open(FILE_NAME, "a") as file:
        file.write(data + "\n")

    print("\nData appended successfully.")

def count_lines():

    if not os.path.exists(FILE_NAME):
        print("\nFile does not exist.")
        return

    with open(FILE_NAME, "r") as file:
        lines = file.readlines()

    print(f"\nTotal Lines = {len(lines)}")

def main():

    while True:

        print("\n" + "=" * 55)
        print("          FILE HANDLING PRACTICE")
        print("=" * 55)

        print("1. Write Data to File")
        print("2. Read File")
        print("3. Append Data")
        print("4. Count Lines")
        print("5. Exit")

        choice = input("\nEnter your choice: ")

        if choice == "1":
            write_file()

        elif choice == "2":
            read_file()

        elif choice == "3":
            append_file()

        elif choice == "4":
            count_lines()

        elif choice == "5":
            print("\nThank you!")
            break

        else:
            print("\nInvalid choice! Please try again.")

if __name__ == "__main__":
    main()                            