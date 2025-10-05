"""Build a similarity network of startups using BERT embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from transformers import BertModel, BertTokenizer

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_DATASET = DATA_DIR / "database.csv"

_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
_model = BertModel.from_pretrained("bert-base-uncased")


@dataclass(frozen=True)
class StartupDataset:
    """Container holding startup metadata."""

    names: np.ndarray
    categories: np.ndarray
    descriptions: np.ndarray

    @staticmethod
    def load(path: Path = DEFAULT_DATASET, limit: int | None = None) -> "StartupDataset":
        """Load startup data from CSV."""

        data = np.genfromtxt(path, delimiter=",", dtype=str, skip_header=1)
        if limit is not None:
            data = data[-limit:]
        return StartupDataset(names=data[:, 0], categories=data[:, 1], descriptions=data[:, 2])


def embed_descriptions(descriptions: Iterable[str]) -> np.ndarray:
    """Compute BERT [CLS] embeddings for ``descriptions``."""

    inputs = _tokenizer(list(descriptions), padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = _model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Return pairwise cosine similarities in ``[0, 1]``."""

    normalised = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarities = normalised @ normalised.T
    return (similarities + 1) / 2  # map [-1, 1] to [0, 1]


def build_similarity_graph(
    dataset: StartupDataset,
    similarity_matrix: np.ndarray,
    threshold: float = 0.7,
) -> nx.Graph:
    """Create an undirected weighted graph connecting similar startups."""

    graph = nx.Graph()
    for idx, (name, category) in enumerate(zip(dataset.names, dataset.categories)):
        graph.add_node(idx, label=name, category=category)
    for i in range(len(dataset.names)):
        for j in range(i + 1, len(dataset.names)):
            weight = similarity_matrix[i, j]
            if weight >= threshold:
                graph.add_edge(i, j, weight=float(weight))
    return graph


def network_metrics(graph: nx.Graph) -> Dict[str, float]:
    """Compute clustering, path length, and modularity scores."""

    metrics: Dict[str, float] = {"clustering": nx.average_clustering(graph)}
    if nx.is_connected(graph):
        metrics["average_path_length"] = nx.average_shortest_path_length(graph)
    else:
        metrics["average_path_length"] = float("nan")
    categories = nx.get_node_attributes(graph, "category")
    communities: Dict[str, set[int]] = {}
    for node, category in categories.items():
        communities.setdefault(category, set()).add(node)
    metrics["modularity"] = nx.community.modularity(graph, list(communities.values()))
    return metrics


def plot_similarity_graph(graph: nx.Graph, similarity_matrix: np.ndarray) -> None:
    """Plot a network layout where edge lengths reflect similarity."""

    layout = nx.spring_layout(graph, weight=None, seed=3)
    plt.figure(figsize=(10, 8))
    colors = [graph.nodes[node]["category"] for node in graph.nodes]
    nx.draw_networkx_nodes(graph, layout, node_size=600, node_color=colors, cmap="tab10")
    nx.draw_networkx_labels(graph, layout)
    weights = [graph[u][v]["weight"] for u, v in graph.edges]
    nx.draw_networkx_edges(graph, layout, width=[w * 2 for w in weights], alpha=0.6)
    plt.axis("off")
    plt.tight_layout()


__all__ = [
    "DEFAULT_DATASET",
    "StartupDataset",
    "build_similarity_graph",
    "cosine_similarity_matrix",
    "embed_descriptions",
    "network_metrics",
    "plot_similarity_graph",
]
