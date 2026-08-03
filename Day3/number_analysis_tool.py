# ==================================================
#            NUMBER ANALYSIS TOOL
#            MLBench Summer Internship DAY 3
# ==================================================

def is_even(number):
    return number % 2 == 0

def is_prime(number):

    if number <= 1:
        return False

    for i in range(2, int(number ** 0.5) + 1):

        if number % i == 0:
            return False

    return True

def count_digits(number):

    number = abs(number)

    if number == 0:
        return 1

    count = 0

    while number > 0:

        count += 1
        number //= 10

    return count

def reverse_number(number):

    original = abs(number)

    reverse = 0

    while original > 0:

        digit = original % 10
        reverse = reverse * 10 + digit
        original //= 10

    return reverse

def is_palindrome(number):

    return abs(number) == reverse_number(number)

def main():

    print("=" * 50)
    print("          NUMBER ANALYSIS TOOL")
    print("=" * 50)

    number = int(input("Enter a Number: "))

    print("\n" + "=" * 50)
    print("             ANALYSIS REPORT")
    print("=" * 50)

    print(f"Number          : {number}")

    if is_even(number):
        print("Even / Odd      : Even")
    else:
        print("Even / Odd      : Odd")

    if is_prime(number):
        print("Prime           : Yes")
    else:
        print("Prime           : No")

    print(f"Digits          : {count_digits(number)}")
    print(f"Reverse         : {reverse_number(number)}")

    if is_palindrome(number):
        print("Palindrome      : Yes")
    else:
        print("Palindrome      : No")

    print("=" * 50)

if __name__ == "__main__":
    main()
        
