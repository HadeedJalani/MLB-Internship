# ==================================================
#        LIBRARY MANAGEMENT SYSTEM
#            MLBench Summer Internship
# ==================================================

import json
import os

FILE_NAME = "Day5/library.json"

class LibraryItem:

    def __init__(self, book_id, title, author):

        self.book_id = book_id
        self.title = title
        self.author = author

class Book(LibraryItem):

    def __init__(self, book_id, title, author, available=True):

        super().__init__(book_id, title, author)

        self.available = available

def load_books():

    if not os.path.exists(FILE_NAME):
        return []

    with open(FILE_NAME, "r") as file:

        data = json.load(file)

    return data["books"]

def save_books(books):

    with open(FILE_NAME, "w") as file:

        json.dump({"books": books}, file, indent=4)

def view_books(books):

    if not books:

        print("\nNo books available.")
        return

    print("\n" + "=" * 60)
    print("              LIBRARY BOOKS")
    print("=" * 60)

    for book in books:

        status = "Available" if book["available"] else "Borrowed"

        print(f"\nID        : {book['book_id']}")
        print(f"Title     : {book['title']}")
        print(f"Author    : {book['author']}")
        print(f"Status    : {status}")

def add_book(books):

    try:

        book_id = int(input("\nEnter Book ID: "))

    except ValueError:

        print("\nInvalid Book ID.")
        return

    for book in books:

        if book["book_id"] == book_id:

            print("\nBook ID already exists.")
            return

    title = input("Enter Book Title: ")
    author = input("Enter Author Name: ")

    new_book = Book(book_id, title, author)

    books.append(new_book.__dict__)

    save_books(books)

    print("\nBook Added Successfully.")

def search_book(books):

    title = input("\nEnter Book Title: ").lower()

    for book in books:

        if book["title"].lower() == title:

            status = "Available" if book["available"] else "Borrowed"

            print("\nBook Found")
            print("-" * 40)
            print(f"ID     : {book['book_id']}")
            print(f"Title  : {book['title']}")
            print(f"Author : {book['author']}")
            print(f"Status : {status}")
            return

    print("\nBook Not Found.")            

def borrow_book(books):

    try:
        book_id = int(input("\nEnter Book ID: "))

    except ValueError:
        print("\nInvalid Book ID.")
        return

    for book in books:

        if book["book_id"] == book_id:

            if book["available"]:

                book["available"] = False
                save_books(books)

                print("\nBook Borrowed Successfully.")

            else:

                print("\nBook is already borrowed.")

            return

    print("\nBook Not Found.")

def return_book(books):

    try:
        book_id = int(input("\nEnter Book ID: "))

    except ValueError:
        print("\nInvalid Book ID.")
        return

    for book in books:

        if book["book_id"] == book_id:

            if not book["available"]:

                book["available"] = True
                save_books(books)

                print("\nBook Returned Successfully.")

            else:

                print("\nBook is already available.")

            return

    print("\nBook Not Found.")

def main():

    books = load_books()

    while True:

        print("\n" + "=" * 60)
        print("          LIBRARY MANAGEMENT SYSTEM")
        print("=" * 60)

        print("1. View All Books")
        print("2. Add Book")
        print("3. Search Book")
        print("4. Borrow Book")
        print("5. Return Book")
        print("6. Exit")

        choice = input("\nEnter your choice: ")

        if choice == "1":
            view_books(books)

        elif choice == "2":
            add_book(books)

        elif choice == "3":
            search_book(books)

        elif choice == "4":
            borrow_book(books)

        elif choice == "5":
            return_book(books)

        elif choice == "6":
            print("\nThank you!")
            break

        else:
            print("\nInvalid Choice! Please try again.")

if __name__ == "__main__":
    main()
                            