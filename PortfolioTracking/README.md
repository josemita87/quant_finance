# Portfolio Tracking

## Overview
Command-line utilities for auditing transaction histories. The `portfolio_tracking` package converts CSV trades into running snapshots, computes cost basis metrics, and generates Matplotlib charts for value and profit breakdowns.

## Setup
```bash
poetry install
```

## Usage
```bash
poetry run python -q - <<'PY'
from pathlib import Path
from portfolio_tracking.portfolio import (
    compute_snapshots,
    load_transactions,
    plot_performance,
    snapshots_to_frame,
)

transactions = load_transactions(Path("transactions.csv"))
snapshots = compute_snapshots(transactions, override_price=42.15)
frame = snapshots_to_frame(snapshots)
plot_performance(frame)
print(frame.tail())
PY
```

## Structure
```
PortfolioTracking/
├── images/
├── plots/
├── src/portfolio_tracking/
│   ├── __init__.py
│   └── portfolio.py
├── transactions.csv
├── pyproject.toml
└── README.md
```

## Highlights
- Parses transaction CSVs with fees, share counts, and trade types
- Running snapshots expose cost basis, unrealised/realised P&L, and total P&L
- Charting helper writes value and P&L plots to `images/`

## Related Projects
- [DataScience](../DataScience/README.md)
- [FinancialMath](../FinancialMath/README.md)
