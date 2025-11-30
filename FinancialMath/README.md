# Financial Math

## Overview
Visualise the efficient frontier and capital market line for a synthetic universe of assets. The `financial_math` package exposes utilities to optimise the tangency portfolio and sample random portfolios for quick demonstrations.

## Quickstart
```bash
uv sync
uv run python -m financial_math.efficient_frontier
```

## Features
- Portfolio statistics helper returning return/volatility/Sharpe
- Sharpe ratio maximisation under long-only constraints
- Random frontier sampling and Matplotlib plotting helpers
- Convenience factory for generating demo data

## Project Layout
```
FinancialMath/
├── src/
│   └── financial_math/
│       ├── __init__.py
│       └── efficient_frontier.py
├── pyproject.toml
└── README.md
```

## Related Projects
- [DataScience](../DataScience/README.md)
- [PortfolioTracking](../PortfolioTracking/README.md)
