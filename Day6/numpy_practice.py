# ==================================================
#           NUMPY PRACTICE - DAY 6
#          MLBench Summer Internship
# ==================================================

import numpy as np


def create_arrays():

    print("\n===== Creating Arrays =====")

    array_1d = np.array([10, 20, 30, 40, 50])

    array_2d = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    print("\n1D Array:")
    print(array_1d)

    print("\n2D Array:")
    print(array_2d)

    return array_1d, array_2d


def array_operations(array):

    print("\n===== Arithmetic Operations =====")

    print("Original Array:")
    print(array)

    print("\nArray + 10")
    print(array + 10)

    print("\nArray * 2")
    print(array * 2)

    print("\nArray / 2")
    print(array / 2)


def array_statistics(array):

    print("\n===== Statistics =====")

    print(f"Maximum : {np.max(array)}")
    print(f"Minimum : {np.min(array)}")
    print(f"Mean    : {np.mean(array)}")
    print(f"Sum     : {np.sum(array)}")


def reshape_array():

    print("\n===== Reshaping Array =====")

    array = np.arange(1, 13)

    print("\nOriginal:")
    print(array)

    reshaped = array.reshape(3, 4)

    print("\nReshaped (3 x 4):")
    print(reshaped)


def slicing_indexing(array):

    print("\n===== Indexing & Slicing =====")

    print("Array:")
    print(array)

    print(f"\nFirst Element: {array[0]}")
    print(f"Last Element : {array[-1]}")

    print("\nFirst Three Elements:")
    print(array[:3])

    print("\nLast Three Elements:")
    print(array[-3:])


def main():

    array_1d, array_2d = create_arrays()

    array_operations(array_1d)

    array_statistics(array_1d)

    reshape_array()

    slicing_indexing(array_1d)


if __name__ == "__main__":
    main()