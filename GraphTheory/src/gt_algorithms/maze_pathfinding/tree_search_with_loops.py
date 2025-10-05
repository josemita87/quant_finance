"""Extended traversal helpers that keep explicit track of explored states."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple

Coordinate = Tuple[int, int]


@dataclass(frozen=True)
class GridWorld:
    """Graph wrapper used to showcase traversals in environments with loops."""

    adjacency: Dict[Coordinate, List[Coordinate]]
    start: Coordinate
    goal: Coordinate

    def neighbours(self, node: Coordinate) -> Iterable[Coordinate]:
        """Yield neighbours for ``node``."""

        yield from self.adjacency.get(node, [])


def iterative_deepening_dfs(world: GridWorld, max_depth: int = 10) -> Optional[List[Coordinate]]:
    """Perform IDDFS to find a path while gracefully handling cycles."""

    def depth_limited(start: Coordinate, limit: int) -> Optional[List[Coordinate]]:
        stack: List[Tuple[Coordinate, int]] = [(start, 0)]
        parents: Dict[Coordinate, Coordinate] = {}
        visited: Set[Coordinate] = set()
        while stack:
            node, depth = stack.pop()
            if node == world.goal:
                return reconstruct_path(parents, world.goal)
            if depth >= limit:
                continue
            visited.add(node)
            for neighbour in world.neighbours(node):
                if neighbour in visited:
                    continue
                parents[neighbour] = node
                stack.append((neighbour, depth + 1))
        return None

    for limit in range(max_depth + 1):
        result = depth_limited(world.start, limit)
        if result:
            return result
    return None


def breadth_first_with_parent_counts(world: GridWorld) -> Optional[List[Coordinate]]:
    """Standard BFS while keeping track of how many times a node is expanded."""

    frontier: Deque[Coordinate] = deque([world.start])
    parents: Dict[Coordinate, Coordinate] = {}
    expansions: Dict[Coordinate, int] = {world.start: 0}

    while frontier:
        node = frontier.popleft()
        if node == world.goal:
            print(f"Expanded {sum(expansions.values())} nodes")
            return reconstruct_path(parents, world.goal)
        expansions[node] = expansions.get(node, 0) + 1
        for neighbour in world.neighbours(node):
            if neighbour in parents or neighbour == world.start:
                continue
            parents[neighbour] = node
            frontier.append(neighbour)
    return None


def reconstruct_path(parents: Dict[Coordinate, Coordinate], goal: Coordinate) -> List[Coordinate]:
    """Reconstruct a path from a ``parent`` dictionary."""

    path = [goal]
    current = goal
    while current in parents:
        current = parents[current]
        path.append(current)
    path.reverse()
    return path


__all__ = [
    "GridWorld",
    "breadth_first_with_parent_counts",
    "iterative_deepening_dfs",
    "reconstruct_path",
]
