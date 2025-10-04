import time
from colorama import init, Fore
from collections import deque
import matplotlib.pyplot as plt

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


# ------------ From Maze I ---------------

# Transformation of the Maze to a Tree
def is_within_bounds(maze, x, y):                       # Function to check if the (x, y) position is within the bounds of the maze
    return 0 <= x < len(maze) and 0 <= y < len(maze[0]) # The position is valid if:
                                                        # - 'x' is between 0 and the number of rows in the maze (len(maze))
                                                        # - 'y' is between 0 and the number of columns in the maze (len(maze[0]))
def is_walkable(maze, x, y):                        # Function to check if the (x, y) position is walkable
    return maze[x][y] == 1 or maze[x][y] == 'B'     # A position is walkable if it contains a '1' (path) or 'B' (exit point)


def tree_creation(maze): # Function to create a tree representation of the maze using DFS (Depth-First Search
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # List of possible movement directions: up, down, left, and right
    root = TreeNode(0, 0, 'A')          # Create the root of the tree at the entrance point of the maze ('A') at position (0, 0)
    visited = set()                                 # Set to keep track of visited nodes so we don't revisit them

    def dfs(node):                  # Inner function that performs Depth-First Search (DFS) recursively
        x, y = node.x, node.y       # Get the current position (x, y) of the node
        visited.add((x, y))         # Mark the current position as visited

        for direction in directions:                             # Loop through all possible directions (up, down, left, right)
            new_x, new_y = x + direction[0], y + direction[1]    # Calculate the new position after moving in the given direction

            if is_within_bounds(maze, new_x, new_y) and is_walkable(maze, new_x, new_y) and (
                    new_x, new_y) not in visited:  # Check if the new position is within the bounds of the maze, is walkable, and has not been visited yet
                value = maze[new_x][new_y]         # Get the value at the new position (either '1' for path or 'B' for exit)
                child_node = TreeNode(new_x, new_y, value)   # Create a new child node at the new position
                node.add_children(child_node)                # Add the child node to the current node
                dfs(child_node)                              # Recursively apply DFS to the child node

    dfs(root)   # Start the DFS from the root node ('A')

    return root # Return the root node of the created tree


# ------------ Maze II -------------------

# Visit all nodes in the Tree starting from A using DFS
# Return the Exit B
def DFS(tree, target = 'B') -> TreeNode:    # We will apply preorder traversal (visit the current node before its children)
    for child in tree.children:             # Loop through all the children of the current node (tree)
        if child.value == target:           # Check if the current child's value matches the target value
            return child                    # If we find the target, return the child node
        node = DFS(child, target)           # Recursively perform DFS on the current child node's subtree to search for 'B'
        if node:                            # If the recursive DFS call found a node (i.e., it's not None)
            return DFS(child, target)       # Call DFS again on the same child and return the result (B node at the end)
                                            # If the target is not found, the function will implicitly return None
# Visit all nodes in the Tree starting from A using BFS
# Return the Exit B
def BFS(Tree, target):              # implementing an iterative approach of BFS
    queue = deque([Tree])           # Initialize a dequeue (double ended queue) with the root node (Tree). BFS uses a queue to explore nodes level by level.
    while queue:                    # While there are still nodes to explore in the queue
        node = queue.popleft()      # Pop (remove) the first node in the queue (FIFO: first-in, first-out)
        if node.value == target:    # If the current node's value matches the target, return the node (target found)
            return node
        queue.extend(node.children) # Add all the current node's children to the end of the queue for further exploration
    return None                     # If the target is not found after exploring all nodes, return None


# Find the path from A to B
def find_path(node):
    path = []                               # We initialize an empty list to store the path from the node to the root
    while node:                             # We traverse up the tree/graph by following the parent pointers until reaching the root (node is None)
        path.append((node.x, node.y))       # We add the current node's coordinates (x, y) to the path
        node = node.parent                  # We move up to the parent node (to continue the traversal upward)
    return path[::-1]                       # We reverse the path because we built it from the target node to the root (we need it from root to target)


# ----------------------------------------
# Main
# ------------ From Maze I ---------------

# Load the Maze
# 0 for the 4x4 maze, 1 for the 6x7 maze, 2 for the 9x9 maze
maze_selection = 2
maze = load_maze(maze_selection)

# Print the maze for visualization (optional)
print_maze(maze)

# Create the root of the tree from the maze, assuming 'A' is at (0,0)
Tree = tree_creation(maze)  # No need for TreeNode as an argument

# Print the tree structure
print_tree(Tree)

# ------------ Maze II -------------------
node_bfs = BFS(Tree, 'B')  # Perform BFS to find the node with value 'B'
node_dfs = DFS(Tree, 'B')  # Perform DFS to find the node with value 'B'
path_bfs = find_path(node_bfs)  # Trace and get the path from 'A' to 'B' for BFS
path_dfs = find_path(node_dfs)  # Trace and get the path from 'A' to 'B' for DFS
# Print the path from A to B
print("\nThe bfs path is:", path_bfs) #Print the bfs path
print("\nThe dfs path is:", path_dfs) #Print the dfs path

def measure_execution_time(algorithm, tree, target):
    start_time = time.perf_counter()  # Record the start time
    algorithm(tree, target)  # Run the specified algorithm (BFS, DFS, etc.)
    end_time = time.perf_counter()  # Record the end time
    return end_time - start_time  # Return the total time taken

# Run 50 trials for BFS and DFS on all three mazes and calculate the average execution time
def run_experiments(iterations = 50):
    bfs_times = [[], [], []]  # BFS times for maze 0, 1, 2
    dfs_times = [[], [], []]  # DFS times for maze 0, 1, 2

    for maze_num in range(3):
        #load maze and create the tree upon which we will run tests
        maze = load_maze(maze_num)  # Load maze from the maze identification number
        tree = tree_creation(maze)  # Create a tree representation of the maze

        # Run BFS and DFS 50 times and record the execution times
        for _ in range(iterations):
            bfs_time = measure_execution_time(BFS, tree, 'B')   # Measure BFS time
            dfs_time = measure_execution_time(DFS, tree, 'B')   # Measure DFS time
            #Keep the result in its according list
            bfs_times[maze_num].append(bfs_time)
            dfs_times[maze_num].append(dfs_time)

    avg_bfs_times = [sum(times) / len(times) for times in bfs_times]    # avarage BFS time
    avg_dfs_times = [sum(times) / len(times) for times in dfs_times]    # avarage DFS time

    return avg_bfs_times, avg_dfs_times

# Plot the average execution times
def plot_results(avg_bfs_times, avg_dfs_times):
    mazes = ['4x4 Maze', '6x7 Maze', '9x9 Maze'] # all mazes we plan to show in the graph on x-axis
    bar_width = 0.35 # lenght of bars
    index = range(len(mazes))

    plt.bar(index, avg_bfs_times, bar_width, label='BFS', color='blue')  #Plotting the bar chart for BFS times
    plt.bar([i + bar_width for i in index], avg_dfs_times, bar_width, label='DFS', color='green') # Plotting a bar chart for DFS times, offset by 'bar_width' to place it next to the BFS bars

    plt.xlabel('Maze (s)')                      #description x axis
    plt.ylabel('Average Time (s)')              #description y axis
    plt.title("Average Execution Time from 50 executions for BFS and DFS in the mazes") #title
    plt.xticks([i + bar_width / 2 for i in index], mazes)   # Set X-ticks for maze labels
    plt.legend()                        # add a legend for identifying  BFS and DFS
    plt.yscale('log')                   # Use logarithmic scale for Y-axis to better visualize large differences
    plt.show()                          # show the plot

# Run experiments and plot the results
avg_bfs_times, avg_dfs_times = run_experiments()    # Get average execution times for BFS and DFS
plot_results(avg_bfs_times, avg_dfs_times)          # Plot the results
