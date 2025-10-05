# Monte Carlo Epidemic Simulation

Implements SI/SIS/SIR simulations on top of NetworkX graphs. Ships as part of the `gt_algorithms` package inside the `GraphTheory` project.

## Features
- Three compartment models (SI, SIS, SIR) with configurable infection/recovery rates.
- Erdős–Rényi graph generator plus helper utilities for normalising adjacency matrices.
- Metrics for clustering, path length, and modularity alongside Matplotlib visualisations.

## Running the Demo
```bash
cd ../../..  # repo root
cd GraphTheory
poetry install
poetry run python -m gt_algorithms.epidemic_simulation.epidemic_simulation
```

## Programmatic Example
```python
from gt_algorithms.epidemic_simulation import (
    EpidemicParameters,
    generate_erdos_renyi_graph,
    initialise_population,
    normalise_by_out_degree,
    simulate_epidemic,
)

adjacency = generate_erdos_renyi_graph(50, 160, seed=7)
normalised = normalise_by_out_degree(adjacency)
states = initialise_population(50, infected_count=3, seed=7)
params = EpidemicParameters(infection_rate=0.3, recovery_rate=0.1, steps=40)
results = simulate_epidemic(normalised, states, params, model="SIR")
print("Peak infected:", results["infected"].max())
```

## Key Parameters
| Symbol | Meaning |
| --- | --- |
| `β` | Infection rate |
| `μ` | Recovery rate |
| `steps` | Simulation horizon |
| `repetitions` | Monte Carlo repetitions when aggregating runs |

The module uses NumPy for vectorised updates and NetworkX/Matplotlib for graph analytics and plotting.
