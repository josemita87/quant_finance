"""Command-line entry points for Monte Carlo portfolio experiments."""

from __future__ import annotations

import argparse
from typing import Callable

import numpy as np

from .monte_carlo_portfolio import (
    ReturnStatistics,
    compute_weekly_returns,
    load_price_data,
    normalise_weights,
    maximum_drawdown_objective,
    sharpe_ratio_objective,
    simulate_portfolio_paths,
    summarise_returns,
    tabu_search,
    value_at_risk_objective,
)

OBJECTIVE_NAMES = ("var", "sharpe", "mdd")


def build_objective(name: str, num_simulations: int) -> Callable[[ReturnStatistics, np.ndarray], float]:
    """Return an objective callable configured with the desired simulation count."""

    if name == "var":
        return lambda stats, weights: value_at_risk_objective(
            stats, weights, num_simulations=num_simulations
        )
    if name == "sharpe":
        return lambda stats, weights: sharpe_ratio_objective(
            stats, weights, num_simulations=num_simulations
        )
    if name == "mdd":
        return lambda stats, weights: maximum_drawdown_objective(
            stats, weights, num_simulations=num_simulations
        )
    raise ValueError(f"Unsupported objective: {name}")


def run_cli() -> None:
    """Execute a simple Monte Carlo optimisation workflow from the command line."""

    parser = argparse.ArgumentParser(description="Optimise a portfolio using Monte Carlo heuristics.")
    parser.add_argument(
        "--objective",
        default="var",
        choices=OBJECTIVE_NAMES,
        help="Portfolio objective to maximise.",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=100_000.0,
        help="Total capital in monetary units.",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=200,
        help="Number of Monte Carlo draws per evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Optional CSV path with historical prices.",
    )
    args = parser.parse_args()

    prices = load_price_data(args.data) if args.data else load_price_data()
    weekly_returns = compute_weekly_returns(prices)
    stats = summarise_returns(weekly_returns)
    rng = np.random.default_rng(args.seed)
    initial_allocation = np.full(stats.mean.size, args.budget / stats.mean.size)

    objective = build_objective(args.objective, args.simulations)

    best_allocation, score = tabu_search(
        stats,
        lambda statistics, weights: objective(
            statistics,
            weights,
        ),
        initial_allocation,
        iterations=60,
        rng=rng,
    )

    weights = normalise_weights(best_allocation)
    simulation = simulate_portfolio_paths(stats, best_allocation, num_simulations=args.simulations, rng=rng)

    print("Optimiser:", "tabu_search")
    print("Objective:", args.objective)
    invested_capital = best_allocation.sum()
    print("Score (monetary units):", score)
    print("Weights (share of capital):", weights)
    print("Simulated mean return (currency):", simulation.mean())
    print("Simulated mean return (% of capital):", simulation.mean() / invested_capital if invested_capital else 0.0)
    print("Simulated volatility (currency):", simulation.std(ddof=1))
    print("Simulated volatility (% of capital):", simulation.std(ddof=1) / invested_capital if invested_capital else 0.0)


if __name__ == "__main__":
    run_cli()
