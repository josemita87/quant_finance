#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Anna, Josep, Miruna, Laura

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

    # Print the results
    print("\n Unsorted array of", size, "elements is:\t\t", random_array)
    print("\n Bubble sorted array of", size, "elements:\t\t", bubble_array)
    print("\n Insertion sorted array of", size, "elements:\t", insertion_array)

    '''
    Our team has developed a dynamic testing and plotting framework to compare algorithm performance, 
    with a focus on modularity. The functions compute_execution_times and plot_graph are defined in 
    W1_searching_template.py for this purpose, allowing flexibility in the testing process. 

    If you prefer a faster, less modular version, let us know, and we can provide it.
    '''

    # We will test the algorithms with different array sizes and measure the execution time.

    # If willing to add other sorting algorithms, add them to the functions_to_test list.
    functions_to_test = (
        (bubblesort, [], "BubbleSort Execution Time"),
        (insertionsort, [], "InsertionSort Execution Time"),
    )

    array_sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    number_range = (1, 100000)

    from W1_searching_template import compute_execution_times, plot_graph

    # Call the function to compute the execution times
    x, y, labels = compute_execution_times(
        functions_to_test, array_sizes, number_range, sorting_function=True
    )

    # Call the function to plot the graph
    plot_graph(x, y, labels)
