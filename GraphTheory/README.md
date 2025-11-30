# Graph Theory

## Overview
Algorithms and simulations covering game search, maze traversal, longest common subsequence, epidemic spread modelling, routing, startup network analysis, and sorting/searching benchmarks. Everything is packaged as importable modules under `gt_algorithms` with runnable examples.

## Installation
```bash
poetry install
```

## Modules
| Module | Highlights |
| --- | --- |
| `gt_algorithms.connect_four` | Minimax with alpha-beta pruning for Connect Four |
| `gt_algorithms.maze_pathfinding` | BFS/DFS + iterative deepening on grid mazes |
| `gt_algorithms.longest_common_subsequence` | Dynamic programming LCS utilities |
| `gt_algorithms.epidemic_simulation` | SI/SIS/SIR Monte Carlo simulations on graphs |
| `gt_algorithms.network_routing` | Random graph generators and Dijkstra helpers |
| `gt_algorithms.sorting_and_searching` | Classic sorting/searching implementations and benchmarks |
| `gt_algorithms.startup_similarity` | BERT-based similarity graph construction |

## Usage Examples
```bash
uv run python -q - <<'PY'
from gt_algorithms.connect_four.minimax_agent import ConnectFourBoard, minimax

board = ConnectFourBoard()
board.drop_disc(3, 'X')
move = minimax(board, depth=4, maximizing_player=True)
print('Suggested column:', move.column)
PY
```

```bash
uv run python -q - <<'PY'
from gt_algorithms.epidemic_simulation import (
    EpidemicParameters,
    generate_erdos_renyi_graph,
    initialise_population,
    normalise_by_out_degree,
    simulate_epidemic,
)

adjacency = generate_erdos_renyi_graph(40, 140, seed=7)
normalised = normalise_by_out_degree(adjacency)
states = initialise_population(40, infected_count=3, seed=7)
params = EpidemicParameters(infection_rate=0.3, recovery_rate=0.1, steps=60)
results = simulate_epidemic(normalised, states, params, model='SIR')
print('Peak infected:', results['infected'].max())
PY
```

```bash
uv run python -q - <<'PY'
from gt_algorithms.startup_similarity.pipeline import (
    StartupDataset,
    build_similarity_graph,
    cosine_similarity_matrix,
    embed_descriptions,
    network_metrics,
)

dataset = StartupDataset.load(limit=25)
embeddings = embed_descriptions(dataset.descriptions)
similarities = cosine_similarity_matrix(embeddings)
graph = build_similarity_graph(dataset, similarities, threshold=0.8)
print(network_metrics(graph))
PY
```

## Structure
```
GraphTheory/
├── assets/startup_similarity/
├── src/gt_algorithms/
│   ├── connect_four/
│   ├── epidemic_simulation/
│   ├── longest_common_subsequence/
│   ├── maze_pathfinding/
│   ├── network_routing/
│   ├── sorting_and_searching/
│   └── startup_similarity/
├── pyproject.toml
└── README.md
```

## Related Projects
- [DataScience](../DataScience/README.md)
- [FinancialMath](../FinancialMath/README.md)
- [MachineLearning](../MachineLearning/README.md)
