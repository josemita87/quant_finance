# Quant Projects Monorepo

## Overview
This workspace curates my quantitative finance, data science, graph theory, and infrastructure experiments in one place. Each project is packaged with Poetry, ships with a focused README, and highlights a distinct skillset that I rely on in day-to-day quantitative engineering work.

## Project Map
| Area | Project | Focus |
| --- | --- | --- |
| Portfolio Analytics | [PortfolioTracking](PortfolioTracking/README.md) | CLI tooling for transaction analysis, cost basis tracking, and reporting |
| Quant Methods | [FinancialMath](FinancialMath/README.md) | Efficient frontier and capital market line exploration |
| Monte Carlo Research | [DataScience](DataScience/README.md) | Portfolio optimisation heuristics built on top of Monte Carlo engines |
| Data Engineering | [DataEngineering](DataEngineering/README.md) | Streaming pipelines (Redpanda, Hopsworks, Docker); untouched ML-course labs |
| Algorithms | [GraphTheory](GraphTheory/README.md) | Graph algorithms, epidemic simulations, startup similarity networks |
| Machine Learning | [MachineLearning](MachineLearning/README.md) | Genetic algorithms on MNIST and other exploratory ML work |

## Getting Started
1. Install Poetry
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
2. Choose a project and install dependencies, e.g.:
   ```bash
   cd GraphTheory
   poetry install
   ```
3. Run the module or open the README inside the project for command examples and datasets.

## Tooling
- Python 3.10+
- Poetry for dependency and environment management
- Ruff for linting/formatting (configured in `ruff.toml`)
- Makefile shortcuts (`make install`, `make format`, `make lint`, `make test`)

## Notes
- PDFs and ad-hoc virtual environments were removed; contextual details now live in the respective READMEs.
- The `DataEngineering/ML-course` directory is preserved exactly as delivered in the course material.
