from colorama import init, Fore
from collections import deque

init()

#Authors: Josep Cubedo, Justin (JJ) Jelinek, Juli Delgado and Martí Alsina
# Definition of the class for the nodes of the tree
# Function inside the class to add children to a node of the tree
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


# Three simple mazes
# 0 is a wall
# 1 is a walkable path
# A is the entrance point
# B is the exit point

def load_maze(selection):
    if (selection == 0):
        map = [['A', 1, 1, 1],
               [0, 1, 0, 0],
               [1, 1, 1, 1],
               [0, 1, 0, 'B']]
    elif (selection == 1):
        map = [['A', 1, 0, 0, 0, 0, 0],
               [0, 1, 1, 0, 0, 'B', 0],
               [0, 1, 0, 0, 1, 1, 0],
               [0, 1, 0, 0, 1, 0, 0],
               [1, 1, 1, 1, 1, 1, 1],
               [1, 0, 0, 0, 1, 0, 1]]
    elif (selection == 2):
        map = [['A', 1, 1, 0, 0, 1, 1, 0, 0],
               [0, 0, 1, 1, 1, 1, 0, 0, 0],
               [1, 1, 1, 0, 0, 1, 1, 0, 0],
               [1, 0, 1, 0, 0, 0, 1, 0, 0],
               [0, 0, 1, 0, 1, 0, 1, 0, 0],
               [1, 0, 1, 0, 1, 0, 1, 1, 'B'],
               [1, 1, 1, 0, 1, 1, 1, 0, 0],
               [1, 0, 0, 0, 0, 0, 1, 0, 1],
               [1, 0, 0, 0, 0, 0, 1, 1, 1]]
    elif (selection == 3):
        map = [['A', 1, 1, 1],
               [0, 1, 0, 1],
               [1, 1, 0, 1],
               [0, 1, 1, 'B']]
    elif (selection == 4):
        map = [['A', 1, 1, 1, 1, 0, 0],
               [1, 0, 0, 0, 1, 1, 'B'],
               [1, 1, 1, 1, 0, 1, 1],
               [1, 0, 0, 1, 0, 0, 0],
               [1, 1, 1, 1, 1, 1, 1],
               [0, 0, 0, 0, 1, 0, 1]]
    elif (selection == 5):
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


# Print the maze as a 2D text matrix
# A is the entrance
# B is the exit
# This function requires colorama library
# to produce colored terminal text and cursor positioning

def print_maze(maze):
    for i in range(0, len(maze)):
        for j in range(0, len(maze[0])):
            if maze[i][j] == 0:
                print(Fore.WHITE, f'{maze[i][j]}', end="")
            elif maze[i][j] == 1:
                print(Fore.GREEN, f'{maze[i][j]}', end="")
            else:
                print(Fore.RED, f'{maze[i][j]}', end="")
        print('\n')
    print(Fore.WHITE, '\n', end="")


# Print the tree structure as a text tree
def print_tree(root, markerStr="+- ", levelMarkers=[]):
    emptyStr = " " * len(markerStr)
    connectionStr = "|" + emptyStr[:-1]
    level = len(levelMarkers)
    mapper = lambda draw: connectionStr if draw else emptyStr
    markers = "".join(map(mapper, levelMarkers[:-1]))
    markers += markerStr if level > 0 else ""
    print(f"{markers}{root.x}", ",", root.y)
    for i, child in enumerate(root.children):
        isLast = i == len(root.children) - 1
        print_tree(child, markerStr, [*levelMarkers, not isLast])


# ------------ From Maze II --------------

# Visit all nodes in the Tree starting from A using DFS
# Return the Exit B
def DFS(Tree, target):      # We will apply preorder traversal (visit the current node before its children)
    for child in Tree.children:             # Loop through all the children of the current node (tree)
        if child.value == target:           # Check if the current child's value matches the target value
            return child                    # If we find the target, return the child node
        node = DFS(child, target)           # Recursively perform DFS on the current child node's subtree to search for 'B'
        if node:                            # If the recursive DFS call found a node (i.e., it's not None)
            return DFS(child, target)       # Call DFS again on the same child and return the result (B node at the end)
                                            # If the target is not found, the function will implicitly return None

def find_path(node):
    path = []                           # We initialize an empty list to store the path from the node to the root
    while node:                         # We traverse up the tree/graph by following the parent pointers until reaching the root (node is None)
        path.append((node.x, node.y))   # We add the current node's coordinates (x, y) to the path
        node = node.parent              # We move up to the parent node (to continue the traversal upward)
    return path[::-1]                   # We reverse the path because we built it from the target node to the root (we need it from root to target)

# ------------ Maze III ---------------

# Transformation of the Maze with possible loops to a Tree
def is_within_bounds(maze, x, y):                       # Function to check if the (x, y) position is within the bounds of the maze
    return 0 <= x < len(maze) and 0 <= y < len(maze[0]) # The position is valid if:
                                                        # 'x' is between 0 and the number of rows in the maze (len(maze))
                                                        # 'y' is between 0 and the number of columns in the maze (len(maze[0]))
def is_walkable(maze, x, y):                        # Function to check if the (x, y) position is walkable
    return maze[x][y] == 1 or maze[x][y] == 'B'     # A position is walkable if it contains a '1' (path) or 'B' (exit point)
def tree_creation_with_loops(maze, node):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # List again possible movement directions: Down, Up, Left, Right
    visited = set()                                   # Set to keep track of visited nodes so we don't revisit them
    queue = deque([node])                             # BFS dequeue (DOUBLE ended queue) with starting with the root

    while queue:                                      # Continue exploring as long as there are nodes in the queue (iterative approach)
        node = queue.popleft()                        # deque the front node from the queue for exploration (FIFO)
        x, y = node.x, node.y                         # Get the current position (x, y) of the node
        visited.add((x, y))                           # Mark the current position as visited
# the combination of an BFS approachand the visited node approach allows for moving multiple ways without getting trapped in a loop. (because already visited)
        for direction in directions:# Explore all possible directions
            new_x, new_y = x + direction[0], y + direction[1]       # get the new position after moving in the given direction

            if is_within_bounds(maze, new_x, new_y) and is_walkable(maze, new_x, new_y) and (new_x, new_y) not in visited: # as in 2 check if visited, walkable and in maze
                value = maze[new_x][new_y]                      # Get the value at the new position (either '1' for path or 'B' for exit)
                child_node = TreeNode(new_x, new_y, value)      # Create a new child node for the walkable path
                node.add_children(child_node)                   # Add the child node to the current node's children

                # Mark the child node as visited and add to queue for further exploration
                visited.add((new_x, new_y))                     # Mark this child as visited
                queue.append(child_node)                        # Enqueue the child node for further exploration


# -------------------------------------
# Main
# ------------ Maze III ---------------

# Load the Maze
# without loop
# 0 for the 4x4 maze, 1 for the 6x7 maze, 2 for the 9x9 maze
# with loops
# 3 for the 4x4 maze, 4 for the 6x7 maze, 5 for the 9x9 maze
maze = load_maze(5)

# Print the maze as a olored text matrix
print_maze(maze)

# Create the root of the tree, which is always 0,0
Tree = TreeNode(0, 0, 'A')
# Create the tree from a maze with possible loops
tree_creation_with_loops(maze, Tree)

# Print the tree
print_tree(Tree)

# ------------ From Maze II --------------

# Call DFS to find the exit B, return the node B
node = DFS(Tree, 'B')

# Call the find_path to return the path from A to B
path = find_path(node)

# Print the path from A to B
print("\nThe path is:", path)
