#!/usr/bin/env python3
"""@authors: Josep, Miruna, Anna, Laura."""

import random
import time
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# We reuse the sorting algorithm from the previous week to sort the array before searching


def sorting_algorithm(unsorted_array: np.ndarray) -> np.ndarray:
    """Return a sorted copy of ``unsorted_array`` using bubble sort."""
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


def linear_search(array: List[int], target: int) -> int:
    """Perform a linear scan to locate ``target``.

    Args:
        array: List of values to inspect.
        target: Value to look for.

    Returns:
        Index of ``target`` if present, otherwise ``-1``.
    """
    for index, value in enumerate(array):
        if value == target:
            return index
    return -1


def binary_search(array: np.ndarray, target: int) -> int:
    """Perform binary search on a sorted 1-D array.

    Args:
        array: Sorted array to search.
        target: Value to look for.

    Returns:
        Index of ``target`` if present, otherwise ``-1``.
    """
    left = 0
    right = len(array) - 1
    while left <= right:
        mid = (left + right) // 2
        if array[mid] == target:
            return mid
        if target < array[mid]:
            right = mid - 1
        else:
            left = mid + 1
    return -1


def binary_search_recursive(array: np.ndarray, target: int, left: int = 0, right: int = -1) -> int:
    """Recursive binary search variant.

    Args:
        array: Sorted array to search.
        target: Value to find.
        left: Current left boundary (inclusive).
        right: Current right boundary (inclusive). ``-1`` defaults to ``len(array) - 1``.

    Returns:
        Index of ``target`` if present, otherwise ``-1``.
    """
    # Set the right index to the last element if not provided (initial iteration)
        if right == -1:
        right = len(array) - 1
    if left > right:
        return -1
    mid = (left + right) // 2
    if array[mid] == target:
        return mid
    if target < array[mid]:
        return binary_search_recursive(array, target, left, mid - 1)
    return binary_search_recursive(array, target, mid + 1, right)


# ----------------------------
# Main
def plot_graph(x_values: List[int], y_values: List[List[float]], y_labels: List[str]) -> None:
    """Plot execution times against array sizes."""
    plt.xlabel("Array size")
    plt.ylabel("CPU execution time (seconds)")
    plt.title("Execution time vs array size")

    # We map the y_values (execution times) to the corresponding plot labels
    for y, label in zip(y_values, y_labels):
        plt.plot(x_values, y, label=label, marker="o")
        plt.legend(loc="upper left")

    plt.show()


def compute_execution_times(
    functions_to_test: List[Tuple[Callable, list, str]],
    array_sizes: List[int],
    number_range: Tuple[int, int],
    sorting_function: bool = False,
) -> Tuple[List[int], List[List[float]], List[str]]:
    """Measure execution times for a collection of callables.

    Args:
        functions_to_test: Tuples of ``(callable, accumulator_list, label)``.
        array_sizes: Sizes to generate when benchmarking.
        number_range: Inclusive range for the random integer generator.
        sorting_function: When ``True`` the tested functions receive the raw array instead of a sorted copy.

    Returns:
        Tuple containing array sizes, list of recorded execution times, and labels for plotting.
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
    searching_benchmarks.py for this purpose, allowing flexibility in the testing process. 

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
