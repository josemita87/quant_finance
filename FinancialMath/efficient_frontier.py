import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco

# Define asset expected returns and covariance matrix
np.random.seed(42)
num_assets = 4
mean_returns = np.random.uniform(0.05, 0.15, num_assets)  # Simulated expected returns
cov_matrix = np.random.rand(num_assets, num_assets)
cov_matrix = (cov_matrix + cov_matrix.T) / 2  # Make it symmetric
np.fill_diagonal(cov_matrix, np.random.uniform(0.02, 0.05, num_assets))  # Variance terms

# Define risk-free rate
risk_free_rate = 0.03

# Function to calculate portfolio statistics
def portfolio_performance(weights, mean_returns, cov_matrix):
    ret = np.sum(weights * mean_returns)
    vol = np.sqrt(weights.T @ cov_matrix @ weights)
    sharpe = (ret - risk_free_rate) / vol
    return ret, vol, sharpe

# Function to minimize (negative Sharpe ratio for max optimization)
def neg_sharpe(weights, mean_returns, cov_matrix):
    return -portfolio_performance(weights, mean_returns, cov_matrix)[2]

# Constraints: sum of weights = 1
constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
bounds = tuple((0, 1) for _ in range(num_assets))
initial_guess = num_assets * [1. / num_assets]

# Optimize for max Sharpe Ratio (Market Portfolio)
optimal_sharpe = sco.minimize(neg_sharpe, initial_guess, args=(mean_returns, cov_matrix),
                              method="SLSQP", bounds=bounds, constraints=constraints)
market_weights = optimal_sharpe.x
market_return, market_vol, market_sharpe = portfolio_performance(market_weights, mean_returns, cov_matrix)

# Generate Efficient Frontier (Random Portfolios)
num_portfolios = 5000
results = np.zeros((3, num_portfolios))

for i in range(num_portfolios):
    weights = np.random.dirichlet(np.ones(num_assets), size=1)[0]  # Random weights summing to 1
    ret, vol, sharpe = portfolio_performance(weights, mean_returns, cov_matrix)
    results[:, i] = [vol, ret, sharpe]

# Plot Efficient Frontier
plt.figure(figsize=(10, 6))
plt.scatter(results[0, :], results[1, :], c=results[2, :], cmap="viridis", marker="o", alpha=0.5)
plt.colorbar(label="Sharpe Ratio")
plt.xlabel("Volatility (Risk)")
plt.ylabel("Expected Return")

# Plot Capital Market Line (CML)
x_cml = np.linspace(0, max(results[0, :]), 100)
y_cml = risk_free_rate + market_sharpe * x_cml
plt.plot(x_cml, y_cml, color="red", linestyle="--", label="Capital Market Line (CML)")

# Plot Market Portfolio (Tangency Portfolio)
plt.scatter(market_vol, market_return, color="red", marker="*", s=200, label="Market Portfolio (M)")

# Plot Risk-Free Rate
plt.scatter(0, risk_free_rate, color="blue", marker="o", s=100, label="Risk-Free Asset")

plt.legend()
plt.title("Efficient Frontier with a Risk-Free Asset (CML)")
plt.show()