# ==================================================
#          PYTHON PRACTICE PROBLEMS - DAY 3
#            MLBench Summer Internship
# ==================================================

# -------------------------------
# CONDITIONAL STATEMENTS
# -------------------------------

def check_positive_negative_zero():
    number = float(input("\nEnter a number: "))

    if number > 0:
        print("The number is Positive.")
    elif number < 0:
        print("The number is Negative.")
    else:
        print("The number is Zero.")


def check_even_odd():
    number = int(input("\nEnter a number: "))

    if number % 2 == 0:
        print("The number is Even.")
    else:
        print("The number is Odd.")


def grade_calculator():
    marks = float(input("\nEnter Marks (0-100): "))

    if marks >= 85:
        grade = "A"
    elif marks >= 80:
        grade = "A-"
    elif marks >= 75:
        grade = "B+"
    elif marks >= 70:
        grade = "B-"
    elif marks >= 65:
        grade = "C+"
    elif marks >= 60:
        grade = "C-"
    elif marks >= 55:
        grade = "D+"
    elif marks >= 50:
        grade = "D-"
    else:
        grade = "F"

    print(f"Grade: {grade}")


def largest_of_three():
    first = float(input("\nEnter First Number: "))
    second = float(input("Enter Second Number: "))
    third = float(input("Enter Third Number: "))

    if first >= second and first >= third:
        largest = first
    elif second >= first and second >= third:
        largest = second
    else:
        largest = third

    print(f"Largest Number: {largest}")


def leap_year():
    year = int(input("\nEnter Year: "))

    if (year % 400 == 0) or (year % 4 == 0 and year % 100 != 0):
        print(f"{year} is a Leap Year.")
    else:
        print(f"{year} is NOT a Leap Year.")

def print_numbers():
    print("\nNumbers from 1 to 100:\n")

    for number in range(1, 101):
        print(number, end=" ")

    print()

def print_even_numbers():
    print("\nEven Numbers from 1 to 100:\n")

    for number in range(2, 101, 2):
        print(number, end=" ")

    print()

def sum_to_n():
    number = int(input("\nEnter a Number: "))

    total = 0

    for i in range(1, number + 1):
        total += i

    print(f"\nSum from 1 to {number} = {total}")

def multiplication_table():
    number = int(input("\nEnter a Number: "))

    print()

    for i in range(1, 11):
        print(f"{number} x {i} = {number * i}")

def count_digits():
    number = abs(int(input("\nEnter a Number: ")))

    digits = len(str(number))

    print(f"\nNumber of Digits = {digits}")

def reverse_number():
    number = int(input("\nEnter a Number: "))

    original = number
    reverse = 0

    while number != 0:
        digit = number % 10
        reverse = reverse * 10 + digit
        number //= 10

    print(f"\nReverse of {original} = {reverse}")

def palindrome():
    number = int(input("\nEnter a Number: "))

    original = number
    reverse = 0

    while number != 0:
        digit = number % 10
        reverse = reverse * 10 + digit
        number //= 10

    if original == reverse:
        print(f"\n{original} is a Palindrome.")
    else:
        print(f"\n{original} is NOT a Palindrome.")

def fibonacci():
    terms = int(input("\nEnter Number of Terms: "))

    first = 0
    second = 1

    print("\nFibonacci Series:")

    for _ in range(terms):
        print(first, end=" ")

        next_number = first + second
        first = second
        second = next_number

    print()

def prime():
    number = int(input("\nEnter a Number: "))

    if number <= 1:
        print(f"\n{number} is NOT a Prime Number.")
        return

    for i in range(2, int(number ** 0.5) + 1):
        if number % i == 0:
            print(f"\n{number} is NOT a Prime Number.")
            return

    print(f"\n{number} is a Prime Number.")

def primes_between():

    print("\nPrime Numbers from 1 to 100:\n")

    for number in range(2, 101):

        is_prime = True

        for divisor in range(2, int(number ** 0.5) + 1):
            if number % divisor == 0:
                is_prime = False
                break

        if is_prime:
            print(number, end=" ")

    print()

def conditional_menu():

    while True:

        print("\n" + "=" * 55)
        print("         CONDITIONAL STATEMENTS")
        print("=" * 55)

        print("1. Positive / Negative / Zero")
        print("2. Even or Odd")
        print("3. Grade Calculator")
        print("4. Largest of Three Numbers")
        print("5. Leap Year Checker")
        print("6. Back to Main Menu")

        choice = input("\nEnter your choice: ")

        if choice == "1":
            check_positive_negative_zero()

        elif choice == "2":
            check_even_odd()

        elif choice == "3":
            grade_calculator()

        elif choice == "4":
            largest_of_three()

        elif choice == "5":
            leap_year()

        elif choice == "6":
            break

        else:
            print("Invalid choice! Please try again.")

def loop_menu():

    while True:

        print("\n" + "=" * 55)
        print("                LOOPS")
        print("=" * 55)

        print("1. Print Numbers (1-100)")
        print("2. Print Even Numbers")
        print("3. Sum from 1 to N")
        print("4. Multiplication Table")
        print("5. Count Digits")
        print("6. Back to Main Menu")

        choice = input("\nEnter your choice: ")

        if choice == "1":
            print_numbers()

        elif choice == "2":
            print_even_numbers()

        elif choice == "3":
            sum_to_n()

        elif choice == "4":
            multiplication_table()

        elif choice == "5":
            count_digits()

        elif choice == "6":
            break

        else:
            print("Invalid choice! Please try again.")

def logic_menu():

    while True:

        print("\n" + "=" * 55)
        print("             LOGIC BUILDING")
        print("=" * 55)

        print("1. Reverse a Number")
        print("2. Palindrome Checker")
        print("3. Fibonacci Series")
        print("4. Prime Number Checker")
        print("5. Prime Numbers (1-100)")
        print("6. Back to Main Menu")

        choice = input("\nEnter your choice: ")

        if choice == "1":
            reverse_number()

        elif choice == "2":
            palindrome()

        elif choice == "3":
            fibonacci()

        elif choice == "4":
            prime()

        elif choice == "5":
            primes_between()

        elif choice == "6":
            break

        else:
            print("Invalid choice! Please try again.")

def main():

    while True:

        print("\n" + "=" * 55)
        print("      PYTHON PRACTICE PROBLEMS")
        print("=" * 55)

        print("1. Conditional Statements")
        print("2. Loops")
        print("3. Logic Building")
        print("4. Exit")

        choice = input("\nEnter your choice: ")

        if choice == "1":
            conditional_menu()

        elif choice == "2":
            loop_menu()

        elif choice == "3":
            logic_menu()

        elif choice == "4":
            print("\nThank you!")
            break

        else:
            print("Invalid choice!")            

if __name__ == "__main__":
    main()
