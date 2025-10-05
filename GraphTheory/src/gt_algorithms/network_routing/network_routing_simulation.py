"""Small toolkit for experimenting with routing strategies on weighted graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


@dataclass(frozen=True)
class RoutingGraph:
    """Wrapper around a NetworkX graph with helper constructors."""

    graph: nx.Graph

    @property
    def adjacency_matrix(self) -> np.ndarray:
        """Return the weighted adjacency matrix."""

        return nx.to_numpy_array(self.graph, weight="weight")

    @staticmethod
    def from_adjacency(matrix: Iterable[Iterable[float]]) -> "RoutingGraph":
        """Create a graph from an adjacency matrix."""

        array = np.array(matrix, dtype=float)
        graph = nx.from_numpy_array(array)
        return RoutingGraph(graph)


PRESET_GRAPHS: Dict[str, RoutingGraph] = {
    "INT9": RoutingGraph.from_adjacency(
        [
            [0, 50, 50, 0, 0, 0, 0, 0, 0],
            [50, 0, 50, 0, 50, 0, 0, 0, 0],
            [50, 50, 0, 50, 0, 0, 0, 0, 0],
            [0, 0, 50, 0, 50, 50, 0, 0, 0],
            [0, 50, 0, 50, 0, 0, 0, 0, 50],
            [0, 0, 0, 50, 0, 0, 50, 50, 50],
            [0, 0, 0, 0, 0, 50, 0, 50, 0],
            [0, 0, 0, 0, 0, 50, 50, 0, 50],
            [0, 0, 0, 0, 50, 50, 0, 50, 0],
        ]
    ),
}


def generate_random_graph(
    num_nodes: int = 10,
    num_edges: int = 20,
    min_weight: int = 1,
    max_weight: int = 10,
    seed: int | None = None,
) -> RoutingGraph:
    """Generate a random weighted graph for experimentation."""

    rng = np.random.default_rng(seed)
    graph = nx.gnm_random_graph(num_nodes, num_edges, seed=rng)
    for u, v in graph.edges:
        graph[u][v]["weight"] = float(rng.integers(min_weight, max_weight + 1))
    return RoutingGraph(graph)


def dijkstra_shortest_path(graph: RoutingGraph, source: int, target: int) -> Tuple[List[int], float]:
    """Return the shortest path and its total weight using Dijkstra's algorithm."""

    path = nx.dijkstra_path(graph.graph, source, target, weight="weight")
    cost = float(nx.dijkstra_path_length(graph.graph, source, target, weight="weight"))
    return path, cost


def plot_graph(graph: RoutingGraph, path: List[int] | None = None) -> None:
    """Visualise a routing graph, highlighting a path if provided."""

    pos = nx.spring_layout(graph.graph, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(graph.graph, pos, node_color="lightblue")
    nx.draw_networkx_labels(graph.graph, pos)
    edge_colors = "lightgray"
    if path and len(path) > 1:
        highlighted_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(graph.graph, pos, edgelist=highlighted_edges, width=2.5, edge_color="tab:red")
        remaining_edges = [edge for edge in graph.graph.edges if edge not in highlighted_edges]
        nx.draw_networkx_edges(graph.graph, pos, edgelist=remaining_edges, edge_color=edge_colors)
    else:
        nx.draw_networkx_edges(graph.graph, pos, edge_color=edge_colors)
    weights = nx.get_edge_attributes(graph.graph, "weight")
    nx.draw_networkx_edge_labels(graph.graph, pos, edge_labels=weights)
    plt.axis("off")
    plt.tight_layout()


__all__ = [
    "PRESET_GRAPHS",
    "RoutingGraph",
    "dijkstra_shortest_path",
    "generate_random_graph",
    "plot_graph",
]
