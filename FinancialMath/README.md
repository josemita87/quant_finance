# Financial Mathematics Module

## Modern Portfolio Theory Implementation

This module implements key concepts from Modern Portfolio Theory (MPT), including the Efficient Frontier and Capital Market Line (CML) calculations.

### Features

- **Efficient Frontier Generation**: Implements Monte Carlo simulation to generate the efficient frontier
- **Market Portfolio Optimization**: Calculates the optimal market portfolio using Sharpe Ratio maximization
- **Capital Market Line**: Plots the CML showing the relationship between risk and return with a risk-free asset
- **Portfolio Performance Metrics**: Calculates expected returns, volatility, and Sharpe ratios

## Implementation Details

### Key Components

1. **Portfolio Performance Calculation**
```python
def portfolio_performance(weights, mean_returns, cov_matrix):
    ret = np.sum(weights * mean_returns)
    vol = np.sqrt(weights.T @ cov_matrix @ weights)
    sharpe = (ret - risk_free_rate) / vol
    return ret, vol, sharpe
```

2. **Optimization Parameters**
- Number of assets: 4
- Risk-free rate: 3%
- Number of random portfolios: 5,000
- Optimization method: Sequential Least Squares Programming (SLSQP)

### Visualization

The module generates a comprehensive plot showing:
- Random portfolio distributions
- Efficient frontier curve
- Capital Market Line
- Market portfolio (tangency point)
- Risk-free asset position

## Dependencies

- NumPy: Array operations and linear algebra
- Matplotlib: Visualization
- SciPy: Optimization routines

## Usage Example

```python
from FinancialMath.efficient_frontier import portfolio_performance

# Define your assets
mean_returns = np.array([...])  # Expected returns
cov_matrix = np.array([...])    # Covariance matrix
weights = np.array([...])       # Portfolio weights

# Calculate portfolio metrics
return_val, volatility, sharpe_ratio = portfolio_performance(weights, mean_returns, cov_matrix)
```

## Mathematical Background

The implementation is based on these key MPT concepts:

1. **Portfolio Return**: \[ R_p = \sum_{i=1}^n w_i R_i \]
2. **Portfolio Volatility**: \[ \sigma_p = \sqrt{w^T \Sigma w} \]
3. **Sharpe Ratio**: \[ SR = \frac{R_p - R_f}{\sigma_p} \]

Where:
- \( w_i \) are the portfolio weights
- \( R_i \) are the expected returns
- \( \Sigma \) is the covariance matrix
- \( R_f \) is the risk-free rate

## Optimization Constraints

- Sum of weights equals 1: \[ \sum_{i=1}^n w_i = 1 \]
- No short-selling: \[ 0 \leq w_i \leq 1 \] for all i

## Notes

- The implementation uses random seed (42) for reproducibility
- The covariance matrix is ensured to be symmetric
- Portfolio weights are generated using the Dirichlet distribution to ensure proper weight constraints 