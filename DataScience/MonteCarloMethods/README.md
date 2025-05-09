# Portfolio Optimization Using Metaheuristics

This repository implements a portfolio optimization framework using **Simulated Annealing (SA)** and **Tabu Search (TS)** metaheuristics. The goal is to enhance portfolio performance across multiple metrics: Value at Risk (VaR), Sharpe Ratio, and Maximum Drawdown (MDD), leveraging Monte Carlo simulations on historical stock price data.

---

## 📊 Features

- Weekly log return transformation from historical daily stock data.
- Monte Carlo simulations of portfolio performance.
- Customizable objective functions: VaR (95%), Sharpe Ratio, MDD.
- Multiple advanced variants of Simulated Annealing and Tabu Search.
- Tabular summary of results for all combinations of methods and metrics.

---

## 🧠 Core Components

### 1. Data Loading and Preprocessing

- `load_data(filepath)`: Loads raw CSV data into a numerical NumPy array.
- `aggregate_data(data)`: Aggregates daily prices into weekly log returns.
- `calculate_mean_std(data)`: Computes per-asset mean and standard deviation of weekly returns.

### 2. Portfolio Initialization

- `initial_solution(size, amount)`: Creates a naive initial portfolio by allocating all capital to a single random asset.

### 3. Portfolio Simulation

- `compute_portfolio_returns(mean, std, solution, num_simulations)`: Uses Monte Carlo sampling to simulate portfolio performance. Includes caching of normal draws for efficiency.

### 4. Objective Functions

- `objective_function_VaR(...)`: Computes 5% VaR.
- `objective_function_sharpe(...)`: Computes Sharpe Ratio (no risk-free rate).
- `objective_function_mdd(...)`: Estimates Maximum Drawdown.

---

## 🔍 Metaheuristics Implemented

### Simulated Annealing Variants

| Variant | Description |
|--------|-------------|
| **v0 Base** | Random capital transfer between assets. |
| **v1 Adaptive Step** | Step size proportional to temperature. |
| **v2 Best-of-K** | Evaluates multiple neighbors per iteration. |
| **v3 Adaptive Reheating** | Reheats temperature after stagnation. |
| **v4 Elitist Archive** | Uses an archive of elite solutions for exploration. |

### Tabu Search Variants

| Variant | Description |
|--------|-------------|
| **v0 Base** | Basic tabu list prevents cycling. |
| **v1 Frequency Memory** | Prioritizes infrequent asset movements. |
| **v2 Candidate List** | Focuses on assets with higher Sharpe ratios. |
| **v3 Aspiration Criterion** | Overrides tabu if solution improves global best. |
| **v4 Random Restart** | Resets from random solution on stagnation. |

---

## 📈 Results Overview

| Metric | Initial | Best SA | Best TS |
|--------|---------|---------|---------|
| **VaR 95%** | 92.60 | 99.03 (SA v2) | 99.16 (TS v3) |
| **Sharpe** | 0.033 | 2.53 (SA v2) | 1.06 (TS v0) |
| **MDD** | -0.115 | -0.0138 (SA v2) | -0.0126 (TS v3) |
| **Exec Time** | — | 0.59s (SA v2) | 0.27s (TS v4) |

---

## ⚙️ Architecture Highlights

- **Modularity**: Functions are atomic and reusable.
- **Efficiency**: Uses NumPy vectorization and random draw caching.
- **Extensibility**: Easily extendable with new objective functions or optimization techniques.

---

## 🛠️ How to Run

```bash
# Install dependencies
pip install numpy pandas

# Run the main experiment loop
python main.py