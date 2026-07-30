# ==================================================
#        OBJECT-ORIENTED PROGRAMMING (OOP)
#            MLBench Summer Internship
# ==================================================

class Student:

    def __init__(self, student_id, name, age):

        self.student_id = student_id
        self.name = name
        self.age = age

    def display(self):

        print("\nStudent Information")
        print("-" * 30)
        print(f"ID   : {self.student_id}")
        print(f"Name : {self.name}")
        print(f"Age  : {self.age}")

student1 = Student(1, "Ali", 20)
student2 = Student(2, "Ahmed", 21)

student1.display()
student2.display()

class Employee:

    def __init__(self, employee_id, name, salary):

        self.employee_id = employee_id
        self.name = name
        self.salary = salary

    def display(self):

        print("\nEmployee Information")
        print("-" * 30)
        print(f"ID     : {self.employee_id}")
        print(f"Name   : {self.name}")
        print(f"Salary : {self.salary}")

employee1 = Employee(101, "Usman", 50000)
employee2 = Employee(102, "Hamza", 65000)

employee1.display()
employee2.display()

class Car:

    def __init__(self, brand, model, year):

        self.brand = brand
        self.model = model
        self.year = year

    def display(self):

        print("\nCar Information")
        print("-" * 30)
        print(f"Brand : {self.brand}")
        print(f"Model : {self.model}")
        print(f"Year  : {self.year}")

    def start(self):

        print(f"{self.brand} {self.model} has started.")

car1 = Car("Toyota", "Corolla", 2022)
car2 = Car("Honda", "Civic", 2023)

car1.display()
car1.start()

car2.display()
car2.start()

students = [

    Student(1, "Ali", 20),
    Student(2, "Ahmed", 21),
    Student(3, "Hadeed", 22)

]

print("\nDisplaying Multiple Students")

for student in students:
    student.display()

# ==================================================
# INHERITANCE
# ==================================================

class Person:

    def __init__(self, name, age):

        self.name = name
        self.age = age

    def display(self):

        print("\nPerson Information")
        print("-" * 30)
        print(f"Name : {self.name}")
        print(f"Age  : {self.age}")

class CollegeStudent(Person):

    def __init__(self, name, age, roll_number):

        super().__init__(name, age)
        self.roll_number = roll_number

    def display(self):

        print("\nStudent Information")
        print("-" * 30)
        print(f"Name        : {self.name}")
        print(f"Age         : {self.age}")
        print(f"Roll Number : {self.roll_number}")

class Teacher(Person):

    def __init__(self, name, age, subject):

        super().__init__(name, age)
        self.subject = subject

    def display(self):

        print("\nTeacher Information")
        print("-" * 30)
        print(f"Name    : {self.name}")
        print(f"Age     : {self.age}")
        print(f"Subject : {self.subject}")

student = CollegeStudent("Hadeed", 22, "CS-221")

teacher = Teacher("Sir Ahmad", 40, "Python")

student.display()
teacher.display()                            

# ==================================================
# ENCAPSULATION
# ==================================================

class BankAccount:

    def __init__(self, owner, balance):

        self.owner = owner
        self.__balance = balance

    def deposit(self, amount):

        self.__balance += amount

    def display_balance(self):

        print(f"\nCurrent Balance: {self.__balance}")

account = BankAccount("Hadeed", 5000)

account.display_balance()

account.deposit(2000)

account.display_balance()        