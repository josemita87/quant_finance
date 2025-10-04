#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@authors: Josep, Miruna, Anna, Laura

"""

import random, time
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Callable

# We reuse the sorting algorithm from the previous week to sort the array before searching


def sorting_algorithm(unsorted_array):

    n = len(unsorted_array)
    sorted_array = np.copy(unsorted_array)

    # Flag to know if the array is sorted
    sorted = False

    # Repeats until the array is sorted
    while not sorted:
        sorted = True

        # Iterate through the array. We use n-1 to avoid indexerrors when reaching endlist
        for i in range(n - 1):
            current = sorted_array[i]

            # If the current element is greater than the next one it swaps them
            if current > sorted_array[i + 1]:
                sorted_array[i] = sorted_array[i + 1]
                sorted_array[i + 1] = current
                sorted = False
            # If no swap occured over the loop, it implies the array is sorted.
            # Therefore, the statement in line 40 will break the while loop.

    # Return the sorted array
    return sorted_array


def linear_search(array: List[int], x: int) -> int:
    """
    Perform a linear search for an element in a list.

    Parameters
    ----------
    array : list of int
        The list to search through.
    x : int
        The element to search for.

    Returns
    -------
    int
        The index of the element if found; otherwise, -1.
    """
    for i, elem in enumerate(array):
        if elem == x:
            return i

    # Return -1 if x is not found
    return -1


def binary_search(array:np.ndarray, x:int) -> int:
    """
    Performs binary search on a sorted array to find the index of the value x.

    Parameters
    ----------
    array : list or numpy.ndarray
        The sorted array to search in.
    x : int or float
        The value to find.

    Returns
    -------
    int
        The index of x in the array if found; otherwise, -1.
    """
    L = 0
    R = len(array) - 1

    while L <= R:
        mid = (L + R) // 2

        if array[mid] == x:
            return mid
        
        #If x is smaller, ignore right half
        elif x < array[mid]:
            R = mid - 1

        #If x is greater, ignore left half
        else:
            L = mid + 1

    # Return -1 if x is not found
    return -1


def binary_search_recursive(array:np.ndarray, x:int, L=0, R=-1) -> int:
    """
    Performs binary search on a sorted array to find the index of the value x.

    Parameters
    ----------
    array : list or numpy.ndarray
        The sorted array to search in.
    x : int or float
        The value to find.
    L : int
        The left index of the search range.
    R : int
        The right index of the search range.

    Returns
    -------
    int
        The index of x in the array if found; otherwise, -1.
    """

    # Set the right index to the last element if not provided (initial iteration)
    if R == -1:
        R = len(array) - 1

    if L > R:
        return -1

    mid = (L + R) // 2

    if array[mid] == x:

        return mid

    elif x < array[mid]:
        return binary_search_recursive(array, x, L, mid - 1)
    else:
        return binary_search_recursive(array, x, mid + 1, R)


# ----------------------------
# Main
def plot_graph(x_values: List[int], y_values: List[List[float]], y_labels: List[str]):

    plt.xlabel("Array size")
    plt.ylabel("CPU execution time (seconds)")
    plt.title("Execution time vs array size")
    
    #We map the y_values (execution times) to the corresponding plot labels
    for y, label in zip(y_values, y_labels):

        plt.plot(x_values, y, label=label, marker="o")
        plt.legend(loc="upper left")

    plt.show()


def compute_execution_times(
    functions_to_test: List[Tuple[Callable, list, str]],
    array_sizes: List[int],
    number_range: Tuple[int, int],
    sorting_function=False,
) -> Tuple[List[int], List[List[float]], List[str]]:
    """
    Computes the execution times of given functions over arrays of different sizes and returns the results.

    Parameters
    ----------
    functions_to_test : List[tuple]
        A list of tuples where each tuple contains a function to test, a list to store its execution times, and a label.
    array_sizes : List[int]
        A list of integers representing the sizes of arrays to test the functions on.
    number_range : Tuple
        A tuple representing the range of numbers to generate the arrays from (inclusive).
    sorting_function : bool, optional
        A flag indicating whether the functions to test are sorting functions. If False, the array will be pre-sorted 
        before testing the functions. Default is False.

    Returns
    -------
    Tuple[List[int], List[List[float]], List[str]]
        A tuple containing the array sizes, a list of lists with execution times for each function, and a list of labels.
    """

    for size in array_sizes:
        # Generate the random array
        array = np.random.randint(*number_range, size=size)

        # Apply sorting algorithm if not testing a sorting function
        if not sorting_function:
            sorted_array = sorting_algorithm(array)
            params = (sorted_array, random.choice(sorted_array))
        # If testing a sorting function, pass the unsorted array
        else:
            params = (array,)

        for function, result, _ in functions_to_test:
            start = time.process_time()
            function(*params)
            end = time.process_time()
            result.append(end - start)

    # Extract execution times for all functions and respective labels
    y_values = [func[1] for func in functions_to_test]
    y_labels = [func[2] for func in functions_to_test]

    # Return the results, so they can be later plotted
    return array_sizes, y_values, y_labels


if __name__ == "__main__":
    # Generate 10 random numbers between 1 and 100
    random_array = np.random.randint(1, 100, size=50)
    size = len(random_array)

    # Sorting the random array
    sorted_array = sorting_algorithm(random_array)

    # Generating a random item to be searched
    x = np.random.randint(0, 100)

    # Execute the linear search returning the position of x or -1 if not present
    idx_ls = linear_search(sorted_array, x)

    if idx_ls != -1:
        print("Item", x, "is present at index ", idx_ls)
    else:
        print("Element", x, "is not present")

    # Execute the binary search returning the position of x or -1 if not present
    idx_bs = binary_search(sorted_array, x)

    if idx_bs != -1:
        print("Item", x, "is present at index ", idx_bs)
    else:
        print("Element", x, "is not present")

    """
    Our team has developed a dynamic testing and plotting framework to compare algorithm performance, 
    with a focus on modularity. The functions compute_execution_times and plot_graph are defined in 
    W1_searching_template.py for this purpose, allowing flexibility in the testing process. 

    If you prefer a faster, less modular version, let us know, and we can provide it.
    """

    # We will test the algorithms with different array sizes and measure the execution time.
    # If willing to add other sorting algorithms, add them to the functions_to_test list.

    functions_to_test = (
        (linear_search, [], "Linear Search Execution Time"),
        (binary_search, [], "Binary Search Execution Time"),
        (binary_search_recursive, [], "Binary Search Recursive Execution Time"),
    )

    array_sizes = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    number_range = (1, 100000)

    # Call the function to compute the execution times
    x, y, labels = compute_execution_times(functions_to_test, array_sizes, number_range)

    # Call the function to plot the graph
    plot_graph(x, y, labels)
