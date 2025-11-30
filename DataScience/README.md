# Data Science

## Overview
Monte Carlo experimentation for portfolio optimisation. The `monte_carlo_methods` package exposes helpers to load price data, summarise returns, and optimise allocations using simulated annealing or tabu search against objectives such as Value at Risk, Sharpe ratio, and maximum drawdown.

## Environment
```bash
uv sync
```

## Usage
```bash
uv run python -m monte_carlo_methods.main --objective var --budget 150000
```

The CLI defaults to a lightweight configuration (200 simulations and 60 tabu iterations) so it completes quickly. Increase `--simulations` or tweak the source in `main.py` for deeper searches.

## Project Structure
```
DataScience/
├── src/
│   └── monte_carlo_methods/
│       ├── __init__.py
│       ├── data/
│       │   └── sp500_prices.csv
│       ├── main.py
│       └── monte_carlo_portfolio.py
├── pyproject.toml
└── README.md
```

## Features
- Weekly return aggregation and summary statistics
- Monte Carlo path generation with reusable RNGs
- Objective helpers for Value at Risk, maximum drawdown, and Sharpe ratio
- Metaheuristic variants: five simulated annealing flavours and five tabu search improvements

To experiment programmatically:

```python
from monte_carlo_methods import (
    load_price_data,
    compute_weekly_returns,
    summarise_returns,
    simulated_annealing_v2_best_of_k,
    value_at_risk_objective,
)

prices = load_price_data()
weekly_returns = compute_weekly_returns(prices)
stats = summarise_returns(weekly_returns)
initial = stats.mean * 100_000
best_allocation, score = simulated_annealing_v2_best_of_k(
    stats,
    lambda s, w: value_at_risk_objective(s, w, num_simulations=500),
    initial,
)
```

## Related Projects
- [FinancialMath](../FinancialMath/README.md)
- [PortfolioTracking](../PortfolioTracking/README.md)
- [GraphTheory](../GraphTheory/README.md)
