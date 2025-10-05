"""Grid-based pathfinding helpers showcasing BFS and DFS traversals."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np

Coordinate = Tuple[int, int]


@dataclass(frozen=True)
class Maze:
    """Simple 2-D maze where 0 denotes walls and 1 denotes walkable tiles."""

    grid: np.ndarray
    start: Coordinate
    goal: Coordinate

    @staticmethod
    def from_list(layout: List[List[object]]) -> "Maze":
        """Create a Maze from a nested list representation."""

        grid = np.array(layout, dtype=object)
        start = tuple(np.argwhere(grid == "A")[0])
        goal = tuple(np.argwhere(grid == "B")[0])
        grid = np.where(grid == "A", 1, np.where(grid == "B", 1, grid)).astype(int)
        return Maze(grid=grid, start=start, goal=goal)

    def neighbours(self, cell: Coordinate) -> Iterable[Coordinate]:
        """Yield walkable neighbours using 4-connectivity."""

        x, y = cell
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid.shape[0] and 0 <= ny < self.grid.shape[1]:
                if self.grid[nx, ny] == 1:
                    yield nx, ny


MAZES: Dict[str, Maze] = {
    "compact": Maze.from_list([["A", 1, 1, 1], [0, 1, 0, 0], [1, 1, 1, 1], [0, 1, 0, "B"]]),
    "medium": Maze.from_list(
        [
            ["A", 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, "B", 0],
            [0, 1, 0, 0, 1, 1, 0],
            [0, 1, 0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1, 0, 1],
        ]
    ),
}


def reconstruct_path(parents: Dict[Coordinate, Coordinate], goal: Coordinate) -> List[Coordinate]:
    """Recreate a path given a ``parent`` back-pointer dictionary."""

    path: List[Coordinate] = [goal]
    current = goal
    while current in parents:
        current = parents[current]
        path.append(current)
    path.reverse()
    return path


def breadth_first_search(maze: Maze) -> Optional[List[Coordinate]]:
    """Find the shortest path using BFS.

    Args:
        maze: Maze describing the search space.

    Returns:
        List of coordinates if a path exists, otherwise ``None``.
    """

    frontier: Deque[Coordinate] = deque([maze.start])
    visited = {maze.start}
    parents: Dict[Coordinate, Coordinate] = {}

    while frontier:
        cell = frontier.popleft()
        if cell == maze.goal:
            return reconstruct_path(parents, maze.goal)
        for neighbour in maze.neighbours(cell):
            if neighbour in visited:
                continue
            visited.add(neighbour)
            parents[neighbour] = cell
            frontier.append(neighbour)
    return None


def depth_first_search(maze: Maze) -> Optional[List[Coordinate]]:
    """Iterative DFS returning the first discovered route to the goal."""

    stack: List[Coordinate] = [maze.start]
    visited = {maze.start}
    parents: Dict[Coordinate, Coordinate] = {}

    while stack:
        cell = stack.pop()
        if cell == maze.goal:
            return reconstruct_path(parents, maze.goal)
        for neighbour in maze.neighbours(cell):
            if neighbour in visited:
                continue
            visited.add(neighbour)
            parents[neighbour] = cell
            stack.append(neighbour)
    return None


__all__ = [
    "MAZES",
    "Maze",
    "breadth_first_search",
    "depth_first_search",
    "reconstruct_path",
]
