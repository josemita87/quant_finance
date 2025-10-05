"""Recursive solution to the Towers of Hanoi puzzle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Move:
    """Representation of a disk transfer between rods."""

    disk: int
    source: str
    destination: str


def solve_hanoi(num_disks: int, source: str, destination: str, auxiliary: str) -> List[Move]:
    """Return the ordered list of moves to solve Towers of Hanoi."""

    if num_disks <= 0:
        return []
    if num_disks == 1:
        return [Move(disk=1, source=source, destination=destination)]
    steps = solve_hanoi(num_disks - 1, source, auxiliary, destination)
    steps.append(Move(disk=num_disks, source=source, destination=destination))
    steps.extend(solve_hanoi(num_disks - 1, auxiliary, destination, source))
    return steps


__all__ = ["Move", "solve_hanoi"]
