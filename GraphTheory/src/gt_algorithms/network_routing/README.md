# Network Routing Utilities

Helpers for generating weighted graphs, computing Dijkstra paths, and visualising results. Distributed as part of the `gt_algorithms` package inside the `GraphTheory` project.

## Features
- Random graph generation (`generate_random_graph`) and preset topologies (`PRESET_GRAPHS`).
- Shortest-path calculation via `dijkstra_shortest_path`.
- Matplotlib visualisations through `plot_graph`.

## Quickstart
```bash
cd ../../..  # repo root
cd GraphTheory
poetry install
poetry run python -q - <<'PY'
from gt_algorithms.network_routing import PRESET_GRAPHS, dijkstra_shortest_path, plot_graph

routing_graph = PRESET_GRAPHS["INT9"]
path, cost = dijkstra_shortest_path(routing_graph, 0, 8)
print("Shortest path", path, "with cost", cost)
plot_graph(routing_graph, path)
PY
```

All dependencies (NetworkX, Matplotlib, NumPy) are declared in `GraphTheory/pyproject.toml`; `poetry install` pulls them in.
