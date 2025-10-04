import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import sklearn.metrics
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Any
import networkx as nx
from matplotlib import cm
from collections import defaultdict

# Load pre-trained BERT model & tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def load_csv(file_path: str, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the latest n rows from csv and returns the startups and their descriptions
    
    Args:
        file_path: Path to the CSV file
        n: Number of rows to load
        
    Returns:
        Tuple of (startups, industry categories, descriptions)
    """
    # Read the entire file
    data = np.genfromtxt(file_path, delimiter=',', dtype=str, skip_header=1)
    
    # Select only the last n rows
    last_n_rows = data[-n:] if len(data) > n else data
    
    return last_n_rows[:, 0], last_n_rows[:, 1], last_n_rows[:, 2]


def word2vec_batch(descriptions: np.ndarray) -> np.ndarray:
    """
    Converts a batch of descriptions into vector embeddings with semantic meaning
    
    Args:
        descriptions: Array of text descriptions
        
    Returns:
        Numpy array of embeddings
    """
    # Tokenize batch descriptions
    inputs = tokenizer(list(descriptions), padding=True, truncation=True, return_tensors='pt')
    
    # Get BERT embeddings (no gradient computation for efficiency)
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract [CLS] token embeddings
    return outputs.last_hidden_state[:, 0, :].numpy()


def normalize(matrix: np.ndarray) -> np.ndarray:
    """
    Normalizes the input matrix to range [0, 1]
    
    Args:
        matrix: Input matrix
        
    Returns:
        Normalized matrix
    """
    min_val, max_val = np.min(matrix), np.max(matrix)
    if max_val == min_val:
        return np.zeros_like(matrix)
    return (matrix - min_val) / (max_val - min_val)


def create_similarity_graph(matrix: np.ndarray, startups: np.ndarray, 
                           categories: np.ndarray, threshold: float) -> nx.Graph:
    """
    Creates a network graph based on the similarity matrix
    
    Args:
        matrix: Similarity matrix
        startups: Array of startup names
        categories: Array of industry categories
        threshold: Threshold for edge creation
        
    Returns:
        NetworkX graph
    """
    G = nx.Graph()
    
    # Add nodes with descriptions and categories
    for idx, (name, category) in enumerate(zip(startups, categories)):
        G.add_node(idx, label=name, category=category)

    # Add edges with weights from the adjacency matrix
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):  # Add edges only for the upper triangle
            if matrix[i, j] > threshold:  # Add edge if similarity is greater than the threshold
                G.add_edge(i, j, weight=matrix[i, j])
                
    return G


def calculate_network_metrics(G: nx.Graph, categories: List[str]) -> Dict[str, float]:
    """
    Calculates key network metrics for the graph
    
    Args:
        G: NetworkX graph
        categories: List of node categories for modularity calculation
        
    Returns:
        Dictionary of network metrics
    """
    metrics = {}
    
    # Calculate average clustering coefficient
    metrics['clustering_coefficient'] = nx.average_clustering(G)
    
    # Calculate average path length (if graph is connected)
    if nx.is_connected(G):
        metrics['average_path_length'] = nx.average_shortest_path_length(G)
    else:
        # For disconnected graphs, calculate average over all connected components
        connected_components = list(nx.connected_components(G))
        path_lengths = []
        for component in connected_components:
            if len(component) > 1:  # Need at least 2 nodes to calculate path length
                subgraph = G.subgraph(component)
                path_lengths.append(nx.average_shortest_path_length(subgraph))
        
        if path_lengths:
            metrics['average_path_length'] = sum(path_lengths) / len(path_lengths)
        else:
            metrics['average_path_length'] = float('nan')  # No valid components
    
    # Calculate modularity based on node categories
    unique_categories = list(set(categories))
    category_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}
    
    # Create community list where each node is assigned its category index
    communities = {}
    for node, data in G.nodes(data=True):
        category = data['category']
        communities[node] = category_to_idx[category]
    
    # Calculate modularity
    metrics['modularity'] = nx.community.modularity(G, [
        {node for node, comm_id in communities.items() if comm_id == i}
        for i in range(len(unique_categories))
    ])
    
    return metrics


def create_similarity_layout(G: nx.Graph, similarity_matrix: np.ndarray, 
                           k: float=0.8, iterations: int=100, 
                           node_size: int=2000) -> Dict[int, np.ndarray]:
    """
    Creates a layout where nodes with higher similarity are positioned closer together,
    while ensuring nodes don't overlap.
    
    Args:
        G: NetworkX graph
        similarity_matrix: Similarity matrix
        k: Optimal distance between nodes (larger values = more spacing)
        iterations: Number of iterations for the spring layout algorithm
        node_size: Size of nodes to calculate minimum distance
        
    Returns:
        Dictionary mapping node ids to positions
    """
    # Set the weight attribute for edges based on similarity
    for i, j in G.edges():
        # Higher similarity = stronger spring = closer nodes
        G[i][j]['weight'] = 1 / (similarity_matrix[i, j] + 0.01)
    
    # Use spring layout with our custom weights
    pos = nx.spring_layout(G, k=k, iterations=iterations, weight='weight')
    
    # Calculate minimum distance between nodes to avoid overlap
    min_dist = np.sqrt(node_size) / 100
    
    # Post-process to prevent node overlap
    for _ in range(50):
        overlap = False
        for i in G.nodes():
            for j in G.nodes():
                if i != j:
                    # Calculate Euclidean distance between nodes
                    dist = np.sqrt((pos[i][0] - pos[j][0])**2 + (pos[i][1] - pos[j][1])**2)
                    
                    # If nodes are too close, push them apart
                    if dist < min_dist:
                        overlap = True
                        # Calculate direction vector
                        direction = np.array([pos[i][0] - pos[j][0], pos[i][1] - pos[j][1]])
                        # Normalize direction vector
                        if np.linalg.norm(direction) > 0:
                            direction = direction / np.linalg.norm(direction)
                        else:
                            # If nodes are exactly in same position, move in random direction
                            direction = np.random.rand(2)
                            direction = direction / np.linalg.norm(direction)
                        
                        # Move nodes slightly apart
                        pos[i] = pos[i] + 0.05 * direction
                        pos[j] = pos[j] - 0.05 * direction
        
        if not overlap:
            break
    
    return pos


def plot_network(G: nx.Graph, similarity_matrix: np.ndarray, pos: Dict = None) -> None:
    """
    Plots a network graph with node colors based on categories and curved edges
    
    Args:
        G: NetworkX graph with 'label' and 'category' node attributes
        similarity_matrix: Similarity matrix
        pos: Optional pre-computed node positions
    """
    if not pos:
        pos = create_similarity_layout(G, similarity_matrix)
    
    plt.figure(figsize=(15, 10))
    
    # Get node labels and weights
    labels = nx.get_node_attributes(G, 'label')
    weights = nx.get_edge_attributes(G, 'weight')
    categories = [G.nodes[node]['category'] for node in G.nodes()]

    # Normalize edge weights for visualization
    if weights:
        min_weight = min(weights.values())
        max_weight = max(weights.values())
        weight_range = max_weight - min_weight
        scaled_weights = {edge: ((weight - min_weight) / weight_range) * 10 for edge, weight in weights.items()}
    else:
        scaled_weights = {}

    # Create color mapping for the categories
    unique_categories = list(set(categories))
    category_to_color = {category: cm.tab10(i) for i, category in enumerate(unique_categories)}

    # Assign node colors based on their category
    node_colors = [category_to_color[G.nodes[node]['category']] for node in G.nodes]

    # Draw edges with curved arrows
    for edge in G.edges():
        i, j = edge
        x1, y1 = pos[i]
        x2, y2 = pos[j]
        
        # Get edge width
        edge_width = scaled_weights.get(edge, 1)
        
        # Draw the edge with curvature
        plt.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-", 
                                    color="gray", 
                                    linewidth=edge_width,
                                    alpha=0.7,
                                    connectionstyle="arc3,rad=0.5"))
    
    # Draw the nodes with category-based colors
    nodes = nx.draw_networkx_nodes(G, pos, node_size=2000, node_color=node_colors, 
                               alpha=0.9, edgecolors='black', linewidths=0.5)
    
    # Ensure nodes are on top of edges
    nodes.set_zorder(2)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold', font_color='black')

    # Create a legend for the categories
    handles = []
    for category, color in category_to_color.items():
        handles.append(plt.Line2D([0], [0], marker='o', color='w', label=category,
                                 markersize=10, markerfacecolor=color))
    plt.legend(handles=handles, title="Categories", loc='best')

    # Finalize plot
    plt.title('Startup Similarity Network', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('startup_network.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_chord_matplotlib(similarity_matrix: np.ndarray, company_names: np.ndarray, threshold: float=0.5) -> None:
    """
    Visualizes relationships as a chord diagram
    
    Args:
        similarity_matrix: Similarity matrix
        company_names: Array of company names
        threshold: Minimum similarity to include in the diagram
    """
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': 'polar'})
    n = len(company_names)

    # Create a circular layout
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)

    for i in range(n):
        ax.bar(theta[i], 1, width=0.02, bottom=0, color='skyblue', edgecolor='black')

    # Add connections based on similarity
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i, j] > threshold:
                angle = (theta[j] - theta[i]) / 2
                x = (theta[i] + theta[j]) / 2
                y = 0.5
                ax.arrow(x, y, np.cos(angle) * 0.3, np.sin(angle) * 0.3,
                         head_width=0.05, head_length=0.1, fc='red', ec='red')

    ax.set_xticks(theta)
    ax.set_xticklabels(company_names, rotation=45, ha="right", fontsize=8)
    ax.set_title("Startup Relationships Diagram", fontsize=16)
    plt.tight_layout()
    plt.savefig('startup_chord.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_heatmap_matplotlib(similarity_matrix: np.ndarray, company_names: np.ndarray) -> None:
    """
    Visualizes the similarity matrix as a heatmap
    
    Args:
        similarity_matrix: Similarity matrix
        company_names: Array of company names
    """
    fig, ax = plt.subplots(figsize=(16, 14))
    
    im = ax.imshow(similarity_matrix, cmap="viridis")
    
    # Add color bar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Similarity Score", rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(company_names)))
    ax.set_yticks(np.arange(len(company_names)))
    ax.set_xticklabels(company_names)
    ax.set_yticklabels(company_names)

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    
    ax.set_title("Startup Similarity Heatmap", fontsize=16)
    fig.tight_layout()
    plt.savefig('startup_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_network_stats(G: nx.Graph, metrics: Dict, startups: np.ndarray, threshold: float) -> None:
    """
    Prints and saves network statistics
    
    Args:
        G: NetworkX graph
        metrics: Dictionary of network metrics
        startups: Array of startup names
        threshold: Similarity threshold used
    """
    # Calculate additional metrics
    density = nx.density(G)
    diameter = nx.diameter(G) if nx.is_connected(G) else "N/A - Graph not connected"
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    avg_degree = sum(degree_sequence) / len(degree_sequence)
    
    # Print metrics to console
    print("\nStartup Network Analysis")
    print("=" * 40)
    print(f"Number of startups: {len(startups)}")
    print(f"Number of connections: {G.number_of_edges()}")
    print(f"Similarity threshold: {threshold:.2f}")
    print("\nNetwork Metrics:")
    print(f"Clustering Coefficient: {metrics['clustering_coefficient']:.2f}")
    print(f"Average Path Length: {metrics['average_path_length']:.2f}")
    print(f"Modularity: {metrics['modularity']:.2f}")
    print(f"Network Density: {density:.4f}")
    print(f"Network Diameter: {diameter}")
    print(f"Average Degree: {avg_degree:.2f}")
    print("=" * 40)
    
    # Save metrics to a file
    with open('network_metrics.txt', 'w') as f:
        f.write("Startup Network Analysis\n")
        f.write("=" * 40 + "\n")
        f.write(f"Number of startups: {len(startups)}\n")
        f.write(f"Number of connections: {G.number_of_edges()}\n")
        f.write(f"Similarity threshold: {threshold:.2f}\n\n")
        f.write("Network Metrics:\n")
        f.write(f"Clustering Coefficient: {metrics['clustering_coefficient']:.2f}\n")
        f.write(f"Average Path Length: {metrics['average_path_length']:.2f}\n")
        f.write(f"Modularity: {metrics['modularity']:.2f}\n")
        f.write(f"Network Density: {density:.4f}\n")
        f.write(f"Network Diameter: {diameter}\n")
        f.write(f"Average Degree: {avg_degree:.2f}\n")
        

def main():
    """Main function to execute the startup network analysis"""
    # Configuration parameters
    n = 37  # Number of startups
    threshold = 0.77  # Similarity threshold
    file_path = 'database.csv'

    # Load data from CSV file
    startups, industry, descr = load_csv(file_path, n)

    # Generate embeddings and similarity matrix
    embeddings = word2vec_batch(descr)
    similarity_matrix = normalize(sklearn.metrics.pairwise.cosine_similarity(embeddings))
    
    # Create network graph
    G = create_similarity_graph(similarity_matrix, startups, industry, threshold)
    
    # Calculate network metrics
    metrics = calculate_network_metrics(G, industry)
    
    # Print and save network statistics
    print_network_stats(G, metrics, startups, threshold)
    
    # Generate layout once and reuse
    pos = create_similarity_layout(G, similarity_matrix)
    
    # Create visualizations
    plot_network(G, similarity_matrix, pos)
    visualize_chord_matplotlib(similarity_matrix, startups, threshold)
    visualize_heatmap_matplotlib(similarity_matrix, startups)


if __name__ == '__main__':
    main()