"""Benchmark helper invoking sorting algorithms and plotting runtime curves."""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

import numpy as np

from .sorting_algorithms import bubblesort, insertionsort
from .searching_benchmarks import compute_execution_times, plot_graph


def benchmark_sorting_algorithms(
    array_sizes: Sequence[int],
    number_range: Tuple[int, int] = (1, 100_000),
) -> None:
    """Benchmark built-in sorting routines and display a performance chart."""

    functions_to_test = (
        (bubblesort, [], "Bubble Sort"),
        (insertionsort, [], "Insertion Sort"),
    )
    x_values, y_values, labels = compute_execution_times(
        list(functions_to_test), list(array_sizes), number_range, sorting_function=True
    )
    plot_graph(x_values, y_values, labels)


__all__ = ["benchmark_sorting_algorithms"]
