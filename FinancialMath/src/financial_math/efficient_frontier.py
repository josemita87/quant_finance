"""Efficient frontier utilities for Modern Portfolio Theory demos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sco


@dataclass(frozen=True)
class PortfolioInputs:
    """Container holding expected returns, covariance matrix, and risk-free rate."""

    mean_returns: np.ndarray
    covariance: np.ndarray
    risk_free_rate: float = 0.0


def portfolio_statistics(
    weights: np.ndarray,
    inputs: PortfolioInputs,
) -> Tuple[float, float, float]:
    """Compute return, volatility, and Sharpe ratio for a weight vector.

    Args:
        weights: Portfolio weights that sum to one.
        inputs: PortfolioInputs with mean returns, covariance, and risk-free rate.

    Returns:
        Tuple containing expected return, volatility, and Sharpe ratio.
    """

    expected_return = float(weights @ inputs.mean_returns)
    volatility = float(np.sqrt(weights.T @ inputs.covariance @ weights))
    sharpe = 0.0 if volatility == 0 else (expected_return - inputs.risk_free_rate) / volatility
    return expected_return, volatility, sharpe


def maximise_sharpe_ratio(
    inputs: PortfolioInputs,
    initial_guess: np.ndarray | None = None,
) -> np.ndarray:
    """Return the weights of the tangency portfolio under a unity-weight constraint.

    Args:
        inputs: PortfolioInputs for the optimisation problem.
        initial_guess: Optional starting point for the solver.

    Returns:
        Optimised weights as a 1-D NumPy array.
    """

    num_assets = inputs.mean_returns.size
    guess = (
        np.full(num_assets, 1 / num_assets)
        if initial_guess is None
        else np.asarray(initial_guess, dtype=float)
    )

    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))

    def negative_sharpe(weights: np.ndarray) -> float:
        _, _, sharpe = portfolio_statistics(weights, inputs)
        return -sharpe

    result = sco.minimize(
        negative_sharpe,
        guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    if not result.success:
        raise RuntimeError(f"Sharpe optimisation failed: {result.message}")
    return result.x


def sample_efficient_frontier(
    inputs: PortfolioInputs,
    num_samples: int = 5000,
    generator: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw random portfolios and compute their statistics.

    Args:
        inputs: PortfolioInputs describing the asset universe.
        num_samples: Number of random portfolios to evaluate.
        generator: Optional random number generator for reproducibility.

    Returns:
        Array with shape ``(num_samples, 3)`` containing volatility, return, Sharpe.
    """

    rng = generator or np.random.default_rng()
    stats = np.zeros((num_samples, 3), dtype=float)
    for idx in range(num_samples):
        weights = rng.dirichlet(np.ones(inputs.mean_returns.size))
        portfolio_return, portfolio_vol, portfolio_sharpe = portfolio_statistics(weights, inputs)
        stats[idx] = (portfolio_return, portfolio_vol, portfolio_sharpe)
    return stats


def plot_efficient_frontier(
    frontier_stats: Iterable[Tuple[float, float, float]],
    tangency_point: Tuple[float, float, float],
    inputs: PortfolioInputs,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Visualise the Monte Carlo efficient frontier and tangency portfolio.

    Args:
        frontier_stats: Iterable of ``(return, volatility, Sharpe)`` tuples.
        tangency_point: Tuple produced by :func:`portfolio_statistics` for the tangency weights.
        inputs: PortfolioInputs used for label formatting.
        ax: Optional Matplotlib axes to draw on.

    Returns:
        Matplotlib Axes with the resulting plot.
    """

    axis = ax or plt.gca()
    returns, volatilities, sharpes = map(np.array, zip(*frontier_stats))
    scatter = axis.scatter(
        volatilities,
        returns,
        c=sharpes,
        cmap="viridis",
        alpha=0.6,
        label="Random Portfolios",
    )
    plt.colorbar(scatter, ax=axis, label="Sharpe Ratio")

    market_vol, market_ret, market_sharpe = tangency_point[1], tangency_point[0], tangency_point[2]
    x_cml = np.linspace(0.0, volatilities.max(), 100)
    y_cml = inputs.risk_free_rate + market_sharpe * x_cml
    axis.plot(x_cml, y_cml, linestyle="--", color="tab:red", label="Capital Market Line")

    axis.scatter(
        market_vol,
        market_ret,
        marker="*",
        s=200,
        color="tab:red",
        label="Tangency Portfolio",
    )
    axis.scatter(
        0.0,
        inputs.risk_free_rate,
        marker="o",
        color="tab:blue",
        label="Risk-Free Asset",
    )
    axis.set_title("Efficient Frontier with Capital Market Line")
    axis.set_xlabel("Volatility (σ)")
    axis.set_ylabel("Expected Return")
    axis.legend()
    return axis


def create_demo_inputs(num_assets: int = 4, seed: int = 42) -> PortfolioInputs:
    """Generate synthetic returns and covariance matrix for quick experimentation."""

    rng = np.random.default_rng(seed)
    mean_returns = rng.uniform(0.05, 0.15, num_assets)
    random_matrix = rng.normal(size=(num_assets, num_assets))
    covariance = np.cov(random_matrix)
    np.fill_diagonal(covariance, rng.uniform(0.02, 0.05, num_assets))
    return PortfolioInputs(mean_returns=mean_returns, covariance=covariance, risk_free_rate=0.03)


__all__ = [
    "PortfolioInputs",
    "create_demo_inputs",
    "maximise_sharpe_ratio",
    "plot_efficient_frontier",
    "portfolio_statistics",
    "sample_efficient_frontier",
]


if __name__ == "__main__":
    inputs = create_demo_inputs()
    tangency_weights = maximise_sharpe_ratio(inputs)
    tangency_stats = portfolio_statistics(tangency_weights, inputs)
    frontier = sample_efficient_frontier(inputs)
    plot_efficient_frontier(frontier, tangency_stats, inputs)
    plt.show()
