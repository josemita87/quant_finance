# Portfolio Optimization Using Metaheuristics

This package powers the Monte Carlo portfolio experiments in the `DataScience` project. It provides helpers to prepare price data, evaluate Value at Risk / maximum drawdown / Sharpe objectives, and explore multiple simulated annealing and tabu-search variants.

## Features
- Weekly-return preprocessing built from `load_price_data`, `compute_weekly_returns`, and `summarise_returns`.
- Monte Carlo path generation via `simulate_portfolio_paths`.
- Objective helpers for VaR, MDD, and Sharpe ratio.
- Five simulated annealing variants and five tabu-search variants with lightweight defaults.

## Optimiser Variants
| Simulated Annealing | Description |
| --- | --- |
| `simulated_annealing_v0_base` | Baseline neighbour sampling |
| `simulated_annealing_v1_adaptive_step` | Slightly slower cooling for wider exploration |
| `simulated_annealing_v2_best_of_k` | Samples multiple neighbours each step |
| `simulated_annealing_v3_adaptive_reheating` | Reheats when progress stalls |
| `simulated_annealing_v4_elitist_archive` | Maintains a small elite archive |

| Tabu Search | Description |
| --- | --- |
| `tabu_search_v0_base` | Baseline tabu strategy |
| `tabu_search_v1_frequency_memory` | Slightly longer tenure and candidate pool |
| `tabu_search_v2_candidate_list` | Larger candidate pool per iteration |
| `tabu_search_v3_aspiration` | Allows tabu overrides when score improves |
| `tabu_search_v4_random_restart` | Periodically restarts from the initial allocation |

## How to Run
```bash
cd ../../..  # repo root
cd DataScience
uv sync
uv run python -m monte_carlo_methods.main --objective var --budget 100000
```

## Programmatic Example
```python
from monte_carlo_methods import (
    load_price_data,
    compute_weekly_returns,
    summarise_returns,
    simulated_annealing_v2_best_of_k,
    value_at_risk_objective,
)

prices = load_price_data()
returns = compute_weekly_returns(prices)
stats = summarise_returns(returns)
initial = stats.mean * 50_000
best_allocation, score = simulated_annealing_v2_best_of_k(
    stats,
    lambda s, w: value_at_risk_objective(s, w, num_simulations=400),
    initial,
)
print("Best VaR score:", score)
```
