# Sorting and Searching Algorithms

This project implements various sorting and searching algorithms in Python. It provides a modular and efficient way to sort and search through data, along with performance testing to compare execution times across different methods.

## Table of Contents

1. [Sorting Algorithms](#sorting-algorithms)
   - [Bubble Sort](#bubble-sort)
   - [Insertion Sort](#insertion-sort)
   - [Quick Sort](#quick-sort)
2. [Searching Algorithms](#searching-algorithms)
   - [Linear Search](#linear-search)
   - [Binary Search](#binary-search)
   - [Binary Search (Recursive)](#binary-search-recursive)
3. [Performance Testing](#performance-testing)
4. [Visualization](#visualization)
5. [Example Usage](#example-usage)

## Sorting Algorithms

### 1. Bubble Sort
The Bubble Sort algorithm works by repeatedly swapping adjacent elements if they are in the wrong order until the array is sorted.

- **Time Complexity**: O(n²)
- **Best Use Case**: Simple and educational, best for small datasets.

### 2. Insertion Sort
Insertion Sort builds a sorted array one element at a time, by repeatedly picking the next element and inserting it into the correct position in the sorted portion.

- **Time Complexity**: O(n²)
- **Best Use Case**: Efficient for small datasets and nearly sorted data.

### 3. Quick Sort
Quick Sort is a divide-and-conquer algorithm that selects a pivot and partitions the array into two halves, recursively sorting each half.

- **Time Complexity**: Average O(n log n), Worst O(n²)
- **Best Use Case**: Efficient for large datasets.

## Searching Algorithms

### 1. Linear Search
A straightforward algorithm that checks each element in the array until the desired element is found.

- **Time Complexity**: O(n)
- **Best Use Case**: Small or unsorted datasets.

### 2. Binary Search
An efficient algorithm that works on sorted arrays, repeatedly dividing the search interval in half to locate the target element.

- **Time Complexity**: O(log n)
- **Best Use Case**: Large sorted datasets.

### 3. Binary Search (Recursive)
A recursive version of the binary search algorithm that utilizes recursion to find the index of the target element.

- **Time Complexity**: O(log n)
- **Best Use Case**: Elegant solution for those who prefer recursion.

## Performance Testing

The project includes a testing framework to measure the execution times of sorting and searching algorithms across various array sizes. The performance testing is conducted using the `compute_execution_times` function, which generates random arrays and records the time taken for each algorithm to process these arrays.

## Visualization

Performance results are visualized using Matplotlib. The execution times for sorting and searching algorithms are plotted against array sizes, providing a clear comparison of their efficiencies. The `plot_graph` function generates this visual representation.

## Example Usage

### Sorting Example
In the main section of the sorting code, a random array of integers is generated and sorted using Bubble Sort, Insertion Sort, and Quick Sort.

```python
if __name__ == "__main__":
    random_array = np.random.randint(1, 10000, size=10)
    bubble_array = bubblesort(random_array)
    insertion_array = insertionsort(random_array)
    quick_array = np.copy(random_array)
    quicksort(quick_array, 0, len(quick_array) - 1)

    print("Unsorted array:", random_array)
    print("Bubble sorted array:", bubble_array)
    print("Insertion sorted array:", insertion_array)
    print("Quick sorted array:", quick_array)