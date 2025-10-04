# Maze Solver with Tree Structure

This project is a Python-based maze solver that creates tree structures to navigate through various mazes. The mazes are defined as grids where cells represent paths, walls, entrances, and exits. The solver employs Depth-First Search (DFS) for pathfinding and uses tree data structures to represent paths within the maze.

## Authors
- Josep Cubedo
- Justin (JJ) Jelinek
- Juli Delgado
- Martí Alsina

## Overview
This project includes:
- Definition of a `TreeNode` class for representing nodes in a tree structure.
- Various functions for maze loading, tree creation, and pathfinding.
- Visualization of mazes in a matrix format using `colorama` for colored terminal output.
- Examples of mazes to test the functionality of the solver.

## Key Components
1. **TreeNode Class**  
   Represents each node within a tree. Each node can store coordinates, a value (representing path, wall, entrance, or exit), and has references to its parent and children nodes.

    ```python
    class TreeNode:
        def __init__(self, x, y, value, parent=None):
            self.x = x
            self.y = y
            self.value = value
            self.parent = parent
            self.children = []

        def add_children(self, *children):
            for child in children:
                child.parent = self
                self.children.append(child)
    ```

2. **Maze Representation**  
   Mazes are represented as 2D lists, where:
   - `0` = Wall
   - `1` = Walkable path
   - `A` = Entrance
   - `B` = Exit

    ```python
    def load_maze(selection):
        if selection == 0:
            map = [['A', 1, 1, 1],
                   [0, 1, 0, 0],
                   [1, 1, 1, 1],
                   [0, 1, 0, 'B']]
        elif selection == 1:
            map = [['A', 1, 0, 0, 0, 0, 0],
                   [0, 1, 1, 0, 0, 'B', 0],
                   [0, 1, 0, 0, 1, 1, 0],
                   [0, 1, 0, 0, 1, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1],
                   [1, 0, 0, 0, 1, 0, 1]]
        elif selection == 2:
            map = [['A', 1, 1, 0, 0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 1, 1, 0, 0, 0],
                   [1, 1, 1, 0, 0, 1, 1, 0, 0],
                   [1, 0, 1, 0, 0, 0, 1, 0, 0],
                   [0, 0, 1, 0, 1, 0, 1, 0, 0],
                   [1, 0, 1, 0, 1, 0, 1, 1, 'B'],
                   [1, 1, 1, 0, 1, 1, 1, 0, 0],
                   [1, 0, 0, 0, 0, 0, 1, 0, 1],
                   [1, 0, 0, 0, 0, 0, 1, 1, 1]]
        elif selection == 3:
            map = [['A', 1, 1, 1],
                   [0, 1, 0, 1],
                   [1, 1, 0, 1],
                   [0, 1, 1, 'B']]
        elif selection == 4:
            map = [['A', 1, 1, 1, 1, 0, 0],
                   [1, 0, 0, 0, 1, 1, 'B'],
                   [1, 1, 1, 1, 0, 1, 1],
                   [1, 0, 0, 1, 0, 0, 0],
                   [1, 1, 1, 1, 1, 1, 1],
                   [0, 0, 0, 0, 1, 0, 1]]
        elif selection == 5:
            map = [['A', 1, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 1, 1, 1, 0, 0, 0],
                   [1, 1, 1, 0, 0, 1, 1, 0, 0],
                   [1, 0, 0, 0, 0, 0, 1, 0, 0],
                   [1, 0, 1, 1, 1, 1, 1, 0, 0],
                   [1, 0, 1, 0, 1, 0, 1, 1, 'B'],
                   [1, 1, 1, 0, 1, 0, 1, 0, 1],
                   [1, 0, 0, 0, 0, 0, 1, 0, 1],
                   [0, 0, 0, 0, 0, 0, 1, 1, 1]]
        return map
    ```

3. **Maze Printing**  
   Prints the maze as a 2D grid. Requires `colorama` for color support.

    ```python
    from colorama import Fore, init
    init(autoreset=True)

    def print_maze(maze):
        for i in range(len(maze)):
            for j in range(len(maze[0])):
                if maze[i][j] == 0:
                    print(Fore.WHITE, f'{maze[i][j]}', end="")
                elif maze[i][j] == 1:
                    print(Fore.GREEN, f'{maze[i][j]}', end="")
                else:
                    print(Fore.RED, f'{maze[i][j]}', end="")
            print('\n')
    ```

4. **Tree Printing**  
   Recursively prints the tree structure, with levels and connections represented textually.

    ```python
    def print_tree(root, markerStr="+- ", levelMarkers=[]):
        emptyStr = " " * len(markerStr)
        connectionStr = "|" + emptyStr[:-1]
        level = len(levelMarkers)
        mapper = lambda draw: connectionStr if draw else emptyStr
        markers = "".join(map(mapper, levelMarkers[:-1]))
        markers += markerStr if level > 0 else ""
        print(f"{markers}{root.x}, {root.y}")
        for i, child in enumerate(root.children):
            isLast = i == len(root.children) - 1
            print_tree(child, markerStr, [*levelMarkers, not isLast])
    ```

5. **Depth-First Search (DFS)**  
   Traverses the tree in a DFS manner to find the exit.

    ```python
    def DFS(tree, target):
        for child in tree.children:
            if child.value == target:
                return child
            node = DFS(child, target)
            if node:
                return node
    ```

6. **Pathfinding**  
   Finds the path from a target node back to the root by tracing parent references.

    ```python
    def find_path(node):
        path = []
        while node:
            path.append((node.x, node.y))
            node = node.parent
        return path[::-1]
    ```

7. **Tree Creation with Loops**  
   Transforms the maze into a tree while allowing loops and avoiding infinite loops with BFS.

    ```python
    from collections import deque

    def is_within_bounds(maze, x, y):
        return 0 <= x < len(maze) and 0 <= y < len(maze[0])

    def is_walkable(maze, x, y):
        return maze[x][y] == 1 or maze[x][y] == 'B'

    def tree_creation_with_loops(maze, node):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        visited = set()
        queue = deque([node])

        while queue:
            node = queue.popleft()
            x, y = node.x, node.y
            visited.add((x, y))

            for direction in directions:
                new_x, new_y = x + direction[0], y + direction[1]

                if is_within_bounds(maze, new_x, new_y) and is_walkable(maze, new_x, new_y) and (new_x, new_y) not in visited:
                    value = maze[new_x