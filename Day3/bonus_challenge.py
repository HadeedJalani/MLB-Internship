# ==================================================
#           LOGIC BUILDING TOOLKIT
#            MLBench Summer Internship DAY 3
# ==================================================

def prime_checker():

    number = int(input("\nEnter a Number: "))

    if number <= 1:
        print("\nNot a Prime Number.")
        return

    for i in range(2, int(number ** 0.5) + 1):

        if number % i == 0:
            print("\nNot a Prime Number.")
            return

    print("\nPrime Number.")

def fibonacci():

    terms = int(input("\nEnter Number of Terms: "))

    first = 0
    second = 1

    print("\nFibonacci Series:\n")

    for _ in range(terms):

        print(first, end=" ")

        next_number = first + second
        first = second
        second = next_number

    print()

def palindrome():

    number = int(input("\nEnter a Number: "))

    original = number
    reverse = 0

    while number > 0:

        digit = number % 10
        reverse = reverse * 10 + digit
        number //= 10

    if original == reverse:
        print("\nPalindrome Number.")
    else:
        print("\nNot a Palindrome.")

def multiplication_table():

    number = int(input("\nEnter a Number: "))

    print()

    for i in range(1, 11):

        print(f"{number} x {i} = {number * i}")

def main():

    while True:

        print("\n" + "=" * 55)
        print("          LOGIC BUILDING TOOLKIT")
        print("=" * 55)

        print("1. Prime Number Checker")
        print("2. Fibonacci Generator")
        print("3. Palindrome Checker")
        print("4. Multiplication Table")
        print("5. Exit")

        choice = input("\nEnter your choice: ")

        if choice == "1":
            prime_checker()

        elif choice == "2":
            fibonacci()

        elif choice == "3":
            palindrome()

        elif choice == "4":
            multiplication_table()

        elif choice == "5":
            print("\nThank you for using the toolkit!")
            break

        else:
            print("\nInvalid Choice! Please try again.")

if __name__ == "__main__":
    main()
    