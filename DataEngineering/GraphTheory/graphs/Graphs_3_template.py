#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 22:55:51 2024
@authors: Josep Cubedo, Julian Harder, Teo Arqués, Gerard Figueras
"""
import copy
from scipy.interpolate import make_interp_spline
import random
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from colorama import init, Fore
init()

#Print the adjacent matrix
def print_matrix( graph ):
    for i in range(0, len(graph)):
        for j in range(0, len(graph[0])):
            if graph[i][j] == 0:
                print(Fore.WHITE, f' {graph[i][j]}', end="")
            elif graph[i][j] > 0:
                print(Fore.GREEN, f' {graph[i][j]}', end="")
            else:
                print(Fore.RED, f'{graph[i][j]}', end="")
        print('\n', end="")
    print(Fore.WHITE, '\n', end="")

#Plot the network
def plot_network( graph, directed = True ):
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    for i in range( len( graph )):
        for j in range( len( graph[0]) ):
            if graph[i][j] != 0:
                G.add_edge( i, j, weight = graph[i][j] )
                # For undirected graphs
                if not directed and not G.has_edge(i,j):
                    G.add_edge(j, i, weight=graph[j][i])

    pos = nx.circular_layout(G)  # positions for all nodes
    # nodes
    nx.draw_networkx_nodes(G, pos)
    # edges
    nx.draw_networkx_edges(G, pos, connectionstyle='arc3, rad = 0.25', arrows=True)
    # node labels
    nx.draw_networkx_labels(G, pos)
    # edge weight labels
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels, connectionstyle='arc3, rad = 0.25')
    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# Graph already created
# INT9 and NSFnet as adjuacent matrix
# Telefonica as adjacent list
def load_graph( selection ):
    if( selection == 'NSFnet'):
         graph = [[  0, 75, 68,  0,  0,  0,  0,100,  0,  0,  0,  0,  0,  0,  0],
                  [ 75,  0, 80, 75,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                  [ 68, 80,  0,  0,  0,  0, 45,  0,  0,  0,  0,  0,  0,  0,  0],
                  [  0, 75,  0,  0, 35,  0,  0,  0,110,  0,  0,  0,  0,  0,  0],
                  [  0,  0,  0, 35,  0, 40, 45,  0,  0,  0,  0,  0,  0,  0,  0],
                  [  0,  0,  0,  0, 40,  0,  0, 25,  0,  0,  0,  0,  0,  0,  0],
                  [  0,  0, 45,  0, 45,  0,  0,  0,  0, 60,  0,  0, 80,  0,  0],
                  [100,  0,  0,  0,  0, 25,  0,  0,  0,  0, 55,  0,  0,  0,  0],
                  [  0,  0,  0,110,  0,  0,  0,  0,  0,  0,  0, 45,  0,100,  0],
                  [  0,  0,  0,  0,  0,  0, 60,  0,  0,  0, 75,  0,  0,  0,  0],
                  [  0,  0,  0,  0,  0,  0,  0, 55,  0, 75,  0, 80,  0, 90, 25],
                  [  0,  0,  0,  0,  0,  0,  0,  0, 45,  0, 80,  0, 75,  0,  0],
                  [  0,  0,  0,  0,  0,  0, 80,  0,  0,  0,  0, 75,  0,110,  0],
                  [  0,  0,  0,  0,  0,  0,  0,  0,100,  0, 90,  0,110,  0,  0],
                  [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 25,  0,  0,  0,  0]]
    elif( selection == 'INT9'):
        graph = [[ 0, 50, 50,  0,  0,  0,  0,  0,  0],
                 [50,  0, 50,  0, 50,  0,  0,  0,  0],
                 [50, 50,  0, 50,  0,  0,  0,  0,  0],
                 [ 0,  0, 50,  0, 50, 50,  0,  0,  0],
                 [ 0, 50,  0, 50,  0,  0,  0,  0, 50],
                 [ 0,  0,  0, 50,  0,  0, 50, 50, 50],
                 [ 0,  0,  0,  0,  0, 50,  0, 50,  0],
                 [ 0,  0,  0,  0,  0, 50, 50,  0, 50],
                 [ 0,  0,  0,  0, 50, 50,  0, 50,  0]]
    elif( selection == 'Telefonica'):
        graph = [[(2,64),(3,64),(4,64)],
                 [(1,64),(3,64),(12,64)],
                 [(1,64),(2,64),(6,64)],
                 [(1,64),(5,64),(6,64)],
                 [(4,64),(8,64),(25,64),(27,64)],
                 [(3,64),(4,64),(9,64),(10,64),(12,64)],
                 [(8,64),(9,64),(20,64),(25,64)],
                 [(5,64),(7,64),(9,64)],
                 [(6,64),(7,64),(8,64),(10,64),(18,64)],
                 [(6,64),(9,64),(11,64),(16,64)],
                 [(10,64),(12,64),(13,64),(16,64)],
                 [(2,64),(6,64),(11,64)],
                 [(11,64),(14,64),(15,64),(16,64)],
                 [(13,64),(15,64),(17,64)],
                 [(13,64),(14,64),(17,64)],
                 [(10,64),(11,64),(13,64),(17,64)],
                 [(14,64),(15,64),(16,64),(18,64),(19,64)],
                 [(9,64),(17,64),(19,64),(20,64)],
                 [(17,64),(18,64),(20,64),(21,64)],
                 [(7,64),(18,64),(19,64),(21,64),(26,64)],
                 [(19,64),(20,64),(22,64)],
                 [(21,64),(23,64),(24,64)],
                 [(22,64),(24,64),(26,64),(28,64)],
                 [(22,64),(23,64),(30,64)],
                 [(5,64),(7,64),(26,64),(27,64),(28,64)],
                 [(20,64),(23,64),(25,64)],
                 [(5,64),(25,64),(28,64),(29,64)],
                 [(23,64),(25,64),(27,64),(29,64),(30,64)],
                 [(27,64),(28,64),(30,64)],
                 [(24,64),(28,64),(29,64)]]
    else:
        print( "Selection is not valid, random graph will be used")
        return None
    return graph

#--------------------------------------------------------------------------
# W1 functions here
#--------------------------------------------------------------------------
import random

def generate_empty_graph(n):
    """
    Generate an empty graph with n nodes, where no nodes are connected.
    
    Parameters:
    - n (int): The number of nodes in the graph.
    
    Returns:
    - List of lists representing the adjacency matrix of the graph.
    """
    return [[0] * n for _ in range(n)]


def get_edge(G, s, d):
    """
    Check if there is an edge between nodes s and d in the graph G.
    
    Parameters:
    - G (list of lists): The adjacency matrix representing the graph.
    - s (int): The source node.
    - d (int): The destination node.
    
    Returns:
    - int or None: The weight of the edge if it exists, otherwise None.
    """
    if G[s][d] != 0:
        return G[s][d]
    return None


def create_edge(G, s, d, w):
    """
    Create an edge between nodes s and d with weight w in the graph G.
    
    Parameters:
    - G (list of lists): The adjacency matrix representing the graph.
    - s (int): The source node.
    - d (int): The destination node.
    - w (int): The weight of the edge.
    
    Returns:
    - List of lists: The updated adjacency matrix.
    """
    G[s][d] = w
    return G


def nodes_selection(G, directed=False):
    """
    Select two different nodes at random from the graph G.
    
    Parameters:
    - G (list of lists): The adjacency matrix representing the graph.
    - directed (bool): Whether the graph is directed (default is False).
    
    Returns:
    - tuple: A tuple of two node indices (source, destination).
    """
    s, d = random.sample(range(len(G)), 2)
    return s, d


def generate_graph(n=10, e=20, min_weight=1, max_weight=10, directed=True):
    """
    Generate a random graph with n nodes and e edges. The graph can be directed or undirected.
    
    Parameters:
    - n (int): The number of nodes.
    - e (int): The number of edges.
    - min_weight (int): The minimum edge weight.
    - max_weight (int): The maximum edge weight.
    - directed (bool): Whether the graph is directed (default is True).
    
    Returns:
    - List of lists: The adjacency matrix representing the graph.
    """
    G = generate_empty_graph(n)  # Start with an empty graph

    edges_set = set()  # To track existing edges and avoid duplicates
    
    for _ in range(e):
        s, d = nodes_selection(G, directed)
        
        # Ensure no duplicate edge exists
        while (s, d) in edges_set or (d, s) in edges_set:
            s, d = nodes_selection(G, directed)
        
        # Assign a random weight to the edge
        w = random.randint(min_weight, max_weight)
        
        create_edge(G, s, d, w)
        edges_set.add((s, d))  # Add directed edge to the set
        
        if not directed:
            create_edge(G, d, s, w)  # For undirected graphs, add reverse edge
            edges_set.add((d, s))  # Add reverse directed edge to the set

    return G


def single_component(G):
    """
    Check if the graph G has a single connected component using BFS.
    
    Parameters:
    - G (list of lists): The adjacency matrix representing the graph.
    
    Returns:
    - bool: True if the graph has a single connected component, otherwise False.
    """
    n = len(G)
    visited = [False] * n
    queue = [0]  # Start BFS from node 0

    visited[0] = True

    while queue:
        node = queue.pop(0)  # Dequeue the first element (FIFO behavior)

        # Visit all neighbors of the current node
        for neighbor in range(n):
            if G[node][neighbor] != 0 and not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)

    return all(visited)  # Check if all nodes were visited


#--------------------------------------------------------------------------
# W2 functions here
#--------------------------------------------------------------------------

def get_edges_list(G):
    """
    Convert the adjacency matrix into a list of edges in the form (source, destination, weight).

    Parameters:
    - G (list of list of int): Adjacency matrix representing the graph.

    Returns:
    - edges (list of tuples): A list of edges, where each edge is represented as a tuple (source, destination, weight).
    """
    
    # List to store edges as (source, destination, weight)
    edges = []

    # Iterate through the adjacency matrix
    for row in range(len(G)):
        for column in range(len(G[row])):
            weight = get_edge(G, row, column)  # Get the edge weight
            
            # Append edge if it exists
            if weight is not None:  # If the edge exists (non-None weight)
                edges.append((row, column, weight))
    
    return edges


def initialize_distance(n, source):
    """
    Initializes the distance array with (node, distance, predecessor) for all nodes in the graph.
    
    Parameters:
    - n (int): The number of nodes in the graph.
    - source (int): The index of the source node.
    
    Returns:
    - distance_array (list): A list of tuples (node, distance, predecessor).
    """
    
    distance_array = []
    
    for i in range(n):
        if i == source:
            distance_array.append((i, 0, None))  # Source node has distance 0 and no predecessor
        else:
            distance_array.append((i, float("inf"), None))  # Other nodes start with infinity distance and no predecessor

    return distance_array


def shortest_paths(G, source):
    """
    Compute the shortest paths from a source node using inverted weights, indicating resource availability.

    Parameters:
    - G: Adjacency matrix or list representing the graph.
    - source: Source node for calculating shortest paths.

    Returns:
    - distances: List of shortest distances from the source to each node.
    - predecessors: List of predecessors for each node in the shortest path.
    """
    number_nodes = len(G)

    # Initialize distance and predecessor arrays
    distance_array = initialize_distance(number_nodes, source)
    edges_list = get_edges_list(G)

    # Relax edges |V| - 1 times
    for _ in range(number_nodes - 1):
        for (src, dest, weight) in edges_list:
            if weight > 0:  # Only process edges with positive weights
                inverted_weight = 1 / weight  # Invert the weight for resource availability
                # Update distance if a shorter path is found
                if distance_array[src][1] + inverted_weight < distance_array[dest][1]:  
                    distance_array[dest] = (dest, distance_array[src][1] + inverted_weight, src)

    # Check for negative-weight cycles
    for (src, dest, weight) in edges_list:
        if weight > 0:
            inverted_weight = 1 / weight
            # If shorter path still exists, a negative cycle is present
            if distance_array[src][1] + inverted_weight < distance_array[dest][1]:
                print("Graph contains a negative-weight cycle")
                return None

    # Extract distances and predecessors for output
    distances = [dist for _, dist, _ in distance_array]
    predecessors = [pred for _, _, pred in distance_array]

    return distances, predecessors


def find_path(prev, s, d):
    """
    Traces the path from source to destination using the predecessor array.

    Parameters:
    - prev (list): Predecessor array where prev[i] is the predecessor of node i in the shortest path.
    - s (int): The source node.
    - d (int): The destination node.

    Returns:
    - path (list): A list of nodes representing the path from source to destination.
                    Returns None if no path exists.
    """
    
    # If source and destination are the same, return the source as the path
    if s == d:
        return [s]
    
    path = []
    current = d
    
    # Trace back from the destination to the source using predecessors
    while current is not None:
        path.append(current)
        if current == s:
            break
        current = prev[current]
    
    # If we did not reach the source, it means there's no path from s to d
    if current != s:
        return None  # No path exists
    
    # Reverse the path to show it from source to destination
    path.reverse()
    return path


#--------------------------------------------------------------------------
# W3 Functions here
#--------------------------------------------------------------------------

def convert_adjacency_list_to_adjacency_matrix(graphL):
    """
    Converts an adjacency list representation of a graph to an adjacency matrix.
    
    Parameters:
    - graphL (list of list of tuples): The adjacency list where each element at index `i`
      contains a list of tuples `(dest, weight)` representing edges from node `i`
      to node `dest` with the given `weight`.
      
    Returns:
    - adj_matrix (list of list of int): The adjacency matrix where `adj_matrix[i][j]` 
      holds the weight of the edge from node `i` to node `j`, or 0 if no edge exists.
    """
    
    # Initialize an adjacency matrix with zeros
    num_nodes = len(graphL)
    adj_matrix = generate_empty_graph(num_nodes)
    
    # Populate the adjacency matrix from the adjacency list
    for node, edges in enumerate(graphL):
        for edge in edges:
            try:
                dest, weight = edge
                adj_matrix[node][dest - 1] = weight  # Adjust for 0-based indexing if necessary
            except IndexError:
                print(f"Error: Node {dest} in adjacency list is out of range for graph size {num_nodes}")
                raise
    
    return adj_matrix


def generate_requests(graph, set_requests):
    """
    Simulate requests in a network and calculate blocked requests based on resource availability.
    If resource availability is 0, then the weight of the edge will equal 'inf' (due to inverted weights).

    Parameters:
    - graph: Adjacency matrix representing the initial network graph.
    - set_requests: Tuple with (start, stop, step) to control the number of requests.

    Returns:
    - requests_array: List of total requests for each iteration.
    - blocked_array: List of blocked requests for each iteration.
    """
    requests_array = []
    blocked_array = []

    # Unpack set_requests parameters
    start, stop, step = set_requests

    # Simulate requests in the graph
    for num_requests in range(start, stop, step):
        # Deep copy of the original graph to reset after each run (avoid cumulative effects)
        current_graph = copy.deepcopy(graph)
        requests_array.append(num_requests)
        
        # Counter for blocked requests in this iteration
        blocked_requests = 0

        for _ in range(num_requests):
            # Randomly select source and destination nodes
            source, destination = nodes_selection(current_graph)
            distances, predecessors = shortest_paths(current_graph, source)

            # Check if destination is reachable
            if distances[destination] == float('inf'):
                blocked_requests += 1
            else:
                path = find_path(predecessors, source, destination)
                if path:
                    current_graph = update_graph(current_graph, path)

        blocked_array.append(blocked_requests)
        
    return requests_array, blocked_array


def update_graph(graph, path):
    """
    Modify the graph by removing resources along the path.

    Parameters:
    - graph: Adjacency matrix representing the graph.
    - path: List of nodes representing the path along which to reduce resources.
    """
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if graph[u][v] > 0:
            graph[u][v] -= 1  # Reduce weight to simulate resource consumption
            graph[v][u] -= 1  # For undirected graphs, update both directions
    return graph


def plot_results(blocked, requests):
    """
    Plot the number of blocked requests against the number of total requests.

    Parameters:
    - blocked: List of blocked requests for each run.
    - requests: List of total requests for each run.
    """
    # Fit a polynomial of degree 4 (or adjust the degree as needed)
    poly_coeffs = np.polyfit(requests, blocked, 4)
    poly_func = np.poly1d(poly_coeffs)

    # Generate smooth x values for the curve
    x_smooth = np.linspace(min(requests), max(requests), 300)
    y_smooth = poly_func(x_smooth)

    # Plot the smoothed trend line
    plt.plot(x_smooth, y_smooth, color='black', label="Trend line")

    # Plot the actual data points
    plt.scatter(requests, blocked, color='red', s=20, label="Data points")

    # Labeling
    plt.xlabel("Number of requests")
    plt.ylabel("Blocked requests")
    plt.title("Results")
    plt.legend()

    plt.show()


#--------------------------------------------------------------------------
#Main
#Variable init
#--------------------------------------------------------------------------
# Week 1 Main
#--------------------------------------------------------------------------

n,e = 10, 25
min_weight, max_weight = 20, 50 
directed = False

#Create random graph
G = generate_graph(n, e, min_weight, max_weight, directed)
while not single_component(G):
    G = generate_graph(n, e, min_weight, max_weight, directed)

print_matrix(G)#Simple print of the adjacent matrix
plot_network(G, directed)#Draw the adjacent matrix

#--------------------------------------------------------------------------
# Week 2 Main
#--------------------------------------------------------------------------

source = 0
destination = 4
dist, prev = shortest_paths(G,source)
if( dist != None ):
    print (f'The distances from the node {source}: {dist}')
    print (f'The shortest path from {source} to {destination}: {find_path(prev, 0, destination)}')
else:
    print ("Shortest path tree cannot be computed due to a negative cycle")

#--------------------------------------------------------------------------
# Week 3 Main
#--------------------------------------------------------------------------

selection = 'INT9'    #INT9, NSFnet, Telefonica, random = other
directed = False

#Load graph
graphL = load_graph(selection)

# Homogenize the graph representation to adjacency matrix
if graphL is None:
    G = generate_graph(n, e, min_weight, max_weight, directed)
    while not single_component(G):
        G = generate_graph(n, e, min_weight, max_weight, directed)

elif selection == 'Telefonica': 
    G = convert_adjacency_list_to_adjacency_matrix(graphL)

else:
    G = graphL

#Draw the adjacent matrix
plot_network(G, directed) 

# Set the requests parameters: initial number of requests,
# the final number of requests and the step between requests
# From 0 requests to 1000 with steps of 50
set_requests = [0, 1000, 50]   


# blocked is an array containing the number of requests blocked due to lack of enough resources
# requests is an array containing the number of requests
# these two arrays are then used to plot the graph requests vs blocked
blocked, requests = generate_requests(G, set_requests)

#Draw the results as a Number of requests vs. Blocked requests
print("Requests Array:", requests)
print("Blocked Array:", blocked)
plot_results(requests, blocked)