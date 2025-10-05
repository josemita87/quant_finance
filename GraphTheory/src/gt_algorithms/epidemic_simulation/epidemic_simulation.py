"""Network-based epidemic simulation utilities for SI/SIS/SIR models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

STATE_SUSCEPTIBLE = "s"
STATE_INFECTED = "i"
STATE_RECOVERED = "r"


@dataclass(frozen=True)
class EpidemicParameters:
    """Parameters controlling epidemic simulations."""

    infection_rate: float
    recovery_rate: float = 0.0
    steps: int = 50


def generate_erdos_renyi_graph(num_nodes: int, num_edges: int, seed: int | None = None) -> np.ndarray:
    """Create an undirected Erdős-Rényi graph as an adjacency matrix."""

    rng = np.random.default_rng(seed)
    possible_edges = num_nodes * (num_nodes - 1) // 2
    num_edges = min(num_edges, possible_edges)
    adjacency = np.zeros((num_nodes, num_nodes), dtype=float)

    candidates = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)]
    rng.shuffle(candidates)
    for i in range(num_edges):
        src, dst = candidates[i]
        adjacency[src, dst] = 1.0
        adjacency[dst, src] = 1.0
    return adjacency


def normalise_by_out_degree(adjacency: np.ndarray) -> np.ndarray:
    """Return a row-normalised adjacency matrix suitable for probability lookups."""

    out_degree = adjacency.sum(axis=1, keepdims=True)
    out_degree[out_degree == 0] = 1
    return adjacency / out_degree


def initialise_population(num_nodes: int, infected_count: int = 1, seed: int | None = None) -> np.ndarray:
    """Create an initial population vector with a subset of infected cases."""

    rng = np.random.default_rng(seed)
    states = np.full(num_nodes, STATE_SUSCEPTIBLE, dtype="U1")
    infected_count = min(infected_count, num_nodes)
    infected_indices = rng.choice(num_nodes, infected_count, replace=False)
    states[infected_indices] = STATE_INFECTED
    return states


def _infection_pressure(adjacency: np.ndarray, states: np.ndarray) -> np.ndarray:
    infected_mask = (states == STATE_INFECTED).astype(float)
    return adjacency @ infected_mask


def si_step(adjacency: np.ndarray, states: np.ndarray, params: EpidemicParameters, rng: np.random.Generator) -> np.ndarray:
    """Execute a single SI model update."""

    pressure = _infection_pressure(adjacency, states)
    infections = (rng.random(len(states)) < params.infection_rate * pressure) & (states == STATE_SUSCEPTIBLE)
    new_states = states.copy()
    new_states[infections] = STATE_INFECTED
    return new_states


def sis_step(
    adjacency: np.ndarray,
    states: np.ndarray,
    params: EpidemicParameters,
    rng: np.random.Generator,
) -> np.ndarray:
    """Execute a single SIS model update."""

    new_states = si_step(adjacency, states, params, rng)
    recoveries = (rng.random(len(states)) < params.recovery_rate) & (new_states == STATE_INFECTED)
    new_states[recoveries] = STATE_SUSCEPTIBLE
    return new_states


def sir_step(
    adjacency: np.ndarray,
    states: np.ndarray,
    params: EpidemicParameters,
    rng: np.random.Generator,
) -> np.ndarray:
    """Execute a single SIR model update."""

    new_states = si_step(adjacency, states, params, rng)
    recoveries = (rng.random(len(states)) < params.recovery_rate) & (new_states == STATE_INFECTED)
    new_states[recoveries] = STATE_RECOVERED
    return new_states


MODEL_STEPS = {
    "SI": si_step,
    "SIS": sis_step,
    "SIR": sir_step,
}


def simulate_epidemic(
    adjacency: np.ndarray,
    initial_states: np.ndarray,
    params: EpidemicParameters,
    model: str = "SIR",
    seed: int | None = None,
) -> Dict[str, np.ndarray]:
    """Run a Monte Carlo simulation of the selected epidemic model.

    Args:
        adjacency: Row-normalised adjacency matrix.
        initial_states: Initial state vector of shape ``(num_nodes,)``.
        params: EpidemicParameters controlling infection dynamics.
        model: One of ``"SI"``, ``"SIS"``, or ``"SIR"``.
        seed: Optional seed for reproducibility.

    Returns:
        Dictionary mapping compartment names to their population proportions over time.
    """

    rng = np.random.default_rng(seed)
    step_fn = MODEL_STEPS[model.upper()]

    states = initial_states.copy()
    susceptible_history = np.zeros(params.steps + 1)
    infected_history = np.zeros(params.steps + 1)
    recovered_history = np.zeros(params.steps + 1)

    def record(step: int, snapshot: np.ndarray) -> None:
        susceptible_history[step] = np.mean(snapshot == STATE_SUSCEPTIBLE)
        infected_history[step] = np.mean(snapshot == STATE_INFECTED)
        recovered_history[step] = np.mean(snapshot == STATE_RECOVERED)

    record(0, states)
    for step in range(1, params.steps + 1):
        states = step_fn(adjacency, states, params, rng)
        record(step, states)

    return {
        "susceptible": susceptible_history,
        "infected": infected_history,
        "recovered": recovered_history,
    }


def plot_compartments(results: Dict[str, np.ndarray]) -> None:
    """Plot S/I/R proportions across simulation steps."""

    plt.figure(figsize=(8, 4))
    steps = np.arange(len(next(iter(results.values()))))
    for label, series in results.items():
        plt.plot(steps, series, label=label.title())
    plt.xlabel("Simulation step")
    plt.ylabel("Population share")
    plt.legend()
    plt.tight_layout()


def project_network(adjacency: np.ndarray) -> nx.Graph:
    """Convert an adjacency matrix into a NetworkX graph for visualisation."""

    graph = nx.from_numpy_array(adjacency)
    return graph


__all__ = [
    "EpidemicParameters",
    "generate_erdos_renyi_graph",
    "initialise_population",
    "normalise_by_out_degree",
    "plot_compartments",
    "project_network",
    "simulate_epidemic",
]
