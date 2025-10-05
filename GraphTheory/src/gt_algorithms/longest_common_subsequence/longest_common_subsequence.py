"""Dynamic programming implementation of the Longest Common Subsequence problem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class LCSResult:
    """Length and sequence returned by :func:`longest_common_subsequence`."""

    length: int
    sequence: str


def longest_common_subsequence(first: str, second: str) -> LCSResult:
    """Compute the longest common subsequence (LCS) between two strings.

    Args:
        first: First input string.
        second: Second input string.

    Returns:
        LCSResult containing both the length and the subsequence itself.
    """

    m, n = len(first), len(second)
    table: List[List[int]] = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if first[i - 1] == second[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])

    sequence: List[str] = []
    i, j = m, n
    while i > 0 and j > 0:
        if first[i - 1] == second[j - 1]:
            sequence.append(first[i - 1])
            i -= 1
            j -= 1
        elif table[i - 1][j] >= table[i][j - 1]:
            i -= 1
        else:
            j -= 1

    sequence.reverse()
    return LCSResult(length=table[m][n], sequence="".join(sequence))


__all__ = ["LCSResult", "longest_common_subsequence"]
