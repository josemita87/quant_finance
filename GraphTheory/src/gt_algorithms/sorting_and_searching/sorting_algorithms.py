#!/usr/bin/env python3
"""@authors: Josep, Miruna, Anna, Laura."""

import numpy as np


def bubblesort(unsorted_array: np.ndarray) -> np.ndarray:
    """Sort an array using the bubble sort algorithm.

    Args:
        unsorted_array: One-dimensional array to sort. The input is not modified.

    Returns:
        NumPy array containing the sorted values.
    """
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


def insertionsort(unsorted_array: np.ndarray) -> np.ndarray:
    """Sort an array using the insertion sort algorithm.

    Args:
        unsorted_array: One-dimensional array to sort. The input is not modified.

    Returns:
        NumPy array containing the sorted values.
    """
    # Create a copy of the unsorted array to sort
    sorted_array = np.copy(unsorted_array)

    # Traverse the array
    for i in range(1, len(sorted_array)):
        key = sorted_array[i]
        j = i - 1

        # Move elements of sorted_array[0..i-1], that are greater than key,
        # to one position ahead of their current position
        while j >= 0 and sorted_array[j] > key:
            sorted_array[j + 1] = sorted_array[j]
            j -= 1
        sorted_array[j + 1] = key

    return sorted_array


# Function to find the partition position in quicksort
def quicksort_partition(array: np.ndarray, left: int, right: int) -> int:
    """Partition ``array`` in-place around a pivot element.

    Args:
        array: Array to partition.
        left: Start index of the region to partition.
        right: End index of the region to partition.

    Returns:
        Index of the pivot after partitioning.
    """
    # Using the middle element as the pivot to avoid worst-case scenarios
    pivot_index = left + (right - left) // 2
    pivot = array[pivot_index]
    array[pivot_index], array[right] = (
        array[right],
        array[pivot_index],
    )  # Move pivot to end
    i = left - 1

    # Partitioning the array
    for j in range(left, right):
        if array[j] <= pivot:
            i += 1
            array[i], array[j] = array[j], array[i]

    # Place pivot in its correct position
    array[i + 1], array[right] = array[right], array[i + 1]
    return i + 1


# Function to perform quicksort
def quicksort(array: np.ndarray, left: int = 0, right: int | None = None) -> np.ndarray:
    """Sort ``array`` in-place using the quicksort algorithm.

    Args:
        array: Array to sort.
        left: Optional lower bound index.
        right: Optional upper bound index. Defaults to the end of ``array``.

    Returns:
        The sorted array (identical object to ``array``).
    """
    if right is None:
        right = len(array) - 1

    if left < right:
        # Partition the array and get the pivot index
        pi = quicksort_partition(array, left, right)
        # Recursively sort the elements before and after partition
        quicksort(array, left, pi - 1)
        quicksort(array, pi + 1, right)

    return array


# ----------------------------
# Main
if __name__ == "__main__":
    # Generate 10 random numbers between 1 and 10000
    random_array = np.random.randint(1, 10000, size=10)
    size = len(random_array)

    # Ordering from smallest to largest using Bubble sort
    bubble_array = bubblesort(random_array)

    # Ordering from smallest to largest using Insertion sort
    insertion_array = insertionsort(random_array)

    # Ordering from smallest to largest using Quick sort
    quick_array = np.copy(random_array)
    quicksort(quick_array, 0, len(quick_array) - 1)

    # Print the results
    print("\n Unsorted array of", size, "elements is:\t\t", random_array)
    print("\n Bubble sorted array of", size, "elements:\t\t", bubble_array)
    print("\n Insertion sorted array of", size, "elements:\t", insertion_array)
    print("\n Quick sorted array of", size, "elements:\t", quick_array)

    """
    Our team has developed a dynamic testing and plotting framework to compare algorithm performance, 
    with a focus on modularity. The functions compute_execution_times and plot_graph are defined in 
    searching_benchmarks.py for this purpose, allowing flexibility in the testing process. 

    If you prefer a faster, less modular version, let us know, and we can provide it.
    """

    from searching_benchmarks import compute_execution_times, plot_graph

    # Define the sort functions to test
    functions_to_test = (
        (bubblesort, [], "BubbleSort Execution Time"),
        (insertionsort, [], "InsertionSort Execution Time"),
        (quicksort, [], "QuickSort Execution Time"),
    )

    array_sizes = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    number_range = (1, 100000)

    # Call the function to compute the execution times
    x, y, labels = compute_execution_times(
        functions_to_test, array_sizes, number_range, sorting_function=True
    )

    # Call the function to plot the graph
    plot_graph(x, y, labels)
