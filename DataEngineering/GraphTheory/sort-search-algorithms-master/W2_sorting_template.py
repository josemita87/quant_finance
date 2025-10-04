#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Josep, Miruna, Anna, Laura

"""

import random, time
import matplotlib.pyplot as plt
import numpy as np


def bubblesort(unsorted_array: np.ndarray) -> np.ndarray:
    """
    Sorts an array using the Bubble sort algorithm.

    The Bubble sort algorithm works by repeatedly swapping the adjacent elements
    if they are in wrong order.

    Parameters
    ----------
    unsorted_array : array_like
        The array to be sorted. The array will not be modified.

    Returns
    -------
    sorted_array : array_like
        A sorted array with the same elements as unsorted_array.
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
    """
    Sorts an array using the Insertion sort algorithm.

    The Insertion sort algorithm works by repeatedly inserting the next element
    from the unsorted part of the array into the sorted part.

    Parameters
    ----------
    unsorted_array : numpy.ndarray
        The array to be sorted. The array will not be modified.

    Returns
    -------
    sorted_array : numpy.ndarray
        A sorted array with the same elements as unsorted_array.
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
def quicksort_partition(array:np.ndarray, left:int, right:int) -> int:
    """
    Partitions the array for the quicksort algorithm.
    This function selects the last element as the pivot and partitions the array such that
    all elements less than or equal to the pivot are on the left of the pivot and all elements
    greater than the pivot are on the right. It then places the pivot in its correct position
    and returns the index of the pivot.
    Args:
        array (list): The list of elements to be partitioned.
        left (int): The starting index of the portion of the array to be partitioned.
        right (int): The ending index of the portion of the array to be partitioned.
    Returns:
        int: The index of the pivot after partitioning.
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
def quicksort(array, left=0, right=None):
    """
    Sorts an array in place using the quicksort algorithm.

    Parameters:
    array (list): The list of elements to be sorted.
    left (int): The starting index of the sublist to be sorted.
    right (int): The ending index of the sublist to be sorted.

    Returns:
    None

    Regarding time complexity, the worst case is O(n^2) this is
    due to the fact that, in the worst case scenario, the pivot
    could be found in one of the ends of the array, making the
    partitioning ineffective. In such case, the algorithm would
    need to iterate through the entire array n times O(n) and
    make n recursive calls O(n).

    Nevertheless, since we have previously handled the pivot
    through the partition function, the average time complexity
    is reduced to O(n log n). This is because the partitioning
    will ensure in each recursion the list is divided in two parts,
    simplifying the process.
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
    W1_searching_template.py for this purpose, allowing flexibility in the testing process. 

    If you prefer a faster, less modular version, let us know, and we can provide it.
    """


    from W1_searching_template import compute_execution_times, plot_graph

    # Define the sort functions to test
    functions_to_test = (
        (bubblesort, [], "BubbleSort Execution Time"),
        (insertionsort, [], "InsertionSort Execution Time"),
        (quicksort, [], "QuickSort Execution Time"),
    )

    array_sizes = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    number_range = (1, 100000)

    from W1_searching_template import compute_execution_times, plot_graph

    # Call the function to compute the execution times
    x, y, labels = compute_execution_times(
        functions_to_test, array_sizes, number_range, sorting_function=True
    )

    # Call the function to plot the graph
    plot_graph(x, y, labels)
