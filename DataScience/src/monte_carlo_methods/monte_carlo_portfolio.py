"""Portfolio optimization helpers for Monte Carlo experimentation.

This module exposes functions to load price data, estimate basic statistics, and run
heuristic optimizers (simulated annealing and tabu search) against Monte Carlo
objectives such as Value at Risk, maximum drawdown, and Sharpe ratio.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_DATA_FILE = DATA_DIR / "sp500_prices.csv"

MonteCarloObjective = Callable[[np.ndarray, np.ndarray, np.ndarray, int], float]


@dataclass(frozen=True)
class ReturnStatistics:
    """Summary statistics for a set of asset returns.

    Attributes:
        mean: Average simple return per asset.
        std: Standard deviation of simple returns per asset.
    """

    mean: np.ndarray
    std: np.ndarray

    @property
    def sharpe_like_scores(self) -> np.ndarray:
        """Return mean-to-volatility scores used for heuristic ordering."""

        denominator = np.where(self.std == 0, np.nan, self.std)
        return self.mean / denominator


def load_price_data(filepath: str | Path = DEFAULT_DATA_FILE) -> np.ndarray:
    """Load adjusted close prices from CSV.

    The CSV is expected to contain a date column followed by one column per ticker.

    Args:
        filepath: Path to a CSV file. Defaults to the bundled S&P 500 sample.

    Returns:
        NumPy array of shape ``(sessions, assets)`` with daily closing prices.

    Raises:
        FileNotFoundError: If the provided file does not exist.
        ValueError: If the file does not contain at least two columns.
    """

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Price file not found: {path}")

    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("CSV must contain a date column followed by asset prices")
    return data[:, 1:]


def compute_weekly_returns(prices: np.ndarray, window: int = 5) -> np.ndarray:
    """Convert daily price levels into simple weekly returns.

    Args:
        prices: Daily prices with shape ``(sessions, assets)``.
        window: Number of trading sessions in a week. Defaults to 5.

    Returns:
        Array containing simple returns for each completed week.

    Raises:
        ValueError: If the input array has fewer rows than ``window + 1``.
    """

    if prices.shape[0] <= window:
        raise ValueError("Not enough sessions to compute weekly returns")

    idx = window * np.arange(1, (prices.shape[0] // window))
    weekly_prices = prices[idx]
    previous_prices = prices[idx - window]
    return (weekly_prices - previous_prices) / previous_prices


def summarise_returns(returns: np.ndarray) -> ReturnStatistics:
    """Calculate mean and standard deviation per asset.

    Args:
        returns: Weekly simple returns with shape ``(weeks, assets)``.

    Returns:
        ReturnStatistics instance with mean and standard deviation vectors.
    """

    return ReturnStatistics(mean=returns.mean(axis=0), std=returns.std(axis=0, ddof=1))


def simulate_portfolio_paths(
    stats: ReturnStatistics,
    weights: np.ndarray,
    num_simulations: int = 200,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw Monte Carlo return samples for a weighted portfolio.

    Args:
        stats: Mean and standard deviation estimates for each asset.
        weights: Allocation vector in monetary units.
        num_simulations: Number of paths to draw.
        rng: Optional NumPy random generator for reproducibility.

    Returns:
        Vector of simulated portfolio returns in monetary units.
    """

    if weights.ndim != 1:
        raise ValueError("weights must be a 1-D vector")

    generator = rng or np.random.default_rng()
    active = weights > 0
    if not np.any(active):
        return np.zeros(num_simulations)

    active_weights = weights[active]
    active_mean = stats.mean[active]
    active_std = stats.std[active]

    draws = generator.normal(loc=0.0, scale=1.0, size=(num_simulations, active_weights.size))
    simulated_asset_returns = active_mean + draws * active_std
    return simulated_asset_returns @ active_weights


def value_at_risk_objective(
    stats: ReturnStatistics,
    weights: np.ndarray,
    num_simulations: int = 200,
    quantile: float = 0.05,
    rng: np.random.Generator | None = None,
) -> float:
    """Return the (1 - quantile) Value at Risk for the portfolio.

    Args:
        stats: Mean and standard deviation per asset.
        weights: Allocation vector.
        num_simulations: Number of Monte Carlo draws.
        quantile: Loss quantile to evaluate (defaults to 5%).
        rng: Optional NumPy random generator.

    Returns:
        The negative-tail percentile expressed in monetary units (higher is better).
    """

    simulated = simulate_portfolio_paths(stats, weights, num_simulations, rng)
    return np.quantile(simulated, quantile)


def maximum_drawdown_objective(
    stats: ReturnStatistics,
    weights: np.ndarray,
    num_simulations: int = 200,
    rng: np.random.Generator | None = None,
) -> float:
    """Estimate the maximum drawdown distribution using Monte Carlo paths.

    Args:
        stats: Mean and standard deviation per asset.
        weights: Allocation vector.
        num_simulations: Number of simulated paths.
        rng: Optional NumPy random generator.

    Returns:
        Worst-case drawdown in percentage terms (closer to zero is better).
    """

    simulated = simulate_portfolio_paths(stats, weights, num_simulations, rng)
    invested = weights.sum()
    if invested == 0:
        return 0.0
    return simulated.min() / invested


def sharpe_ratio_objective(
    stats: ReturnStatistics,
    weights: np.ndarray,
    num_simulations: int = 200,
    rng: np.random.Generator | None = None,
) -> float:
    """Estimate the Sharpe ratio from simulated returns.

    Args:
        stats: Mean and standard deviation per asset.
        weights: Allocation vector.
        num_simulations: Number of simulated paths.
        rng: Optional NumPy random generator.

    Returns:
        Monte Carlo estimate of the Sharpe ratio assuming a zero risk-free rate.
    """

    simulated = simulate_portfolio_paths(stats, weights, num_simulations, rng)
    std = simulated.std(ddof=1)
    return 0.0 if std == 0 else simulated.mean() / std


def normalise_weights(weights: np.ndarray) -> np.ndarray:
    """Transform a monetary allocation into weights that sum to the invested capital."""

    total = weights.sum()
    return weights if total == 0 else weights / total




def _random_neighbour(
    allocation: np.ndarray,
    rng: np.random.Generator,
    invested: np.ndarray,
    best_of_k: int = 1,
) -> np.ndarray:
    """Return a neighbouring allocation using random fund transfers."""

    best_candidate = allocation
    best_transfer = -np.inf
    for _ in range(best_of_k):
        neighbour = allocation.copy()
        source = int(rng.choice(invested)) if invested.size else int(rng.integers(0, allocation.size))
        target = int(rng.integers(0, allocation.size))
        if source == target and allocation.size > 1:
            target = (target + 1) % allocation.size
        transfer = allocation[source] * rng.random()
        neighbour[source] -= transfer
        neighbour[target] += transfer
        if transfer > best_transfer:
            best_candidate = neighbour
            best_transfer = transfer
    return best_candidate


def _simulated_annealing_core(
    stats: ReturnStatistics,
    objective: Callable[[ReturnStatistics, np.ndarray], float],
    initial_allocation: np.ndarray,
    rng: np.random.Generator,
    initial_temperature: float,
    cooling_rate: float,
    inner_iterations: int,
    best_of_k: int = 1,
    reheating_patience: int | None = None,
    reheating_factor: float = 1.2,
    archive_size: int = 0,
) -> tuple[np.ndarray, float]:
    """Shared simulated annealing implementation powering all variants."""

    current = initial_allocation.astype(float)
    current_score = objective(stats, current)
    best = current.copy()
    best_score = current_score
    temperature = max(initial_temperature, 1e-3)
    stagnation = 0
    archive: list[tuple[np.ndarray, float]] = []

    while temperature > 1e-3:
        invested = np.where(current > 0)[0]
        for _ in range(inner_iterations):
            neighbour = _random_neighbour(current, rng, invested, best_of_k=best_of_k)
            neighbour_score = objective(stats, neighbour)
            delta = neighbour_score - current_score
            if delta > 0 or rng.random() < np.exp(delta / temperature):
                current = neighbour
                current_score = neighbour_score
                if current_score > best_score:
                    best = current.copy()
                    best_score = current_score
                    stagnation = 0
                else:
                    stagnation += 1
            else:
                stagnation += 1

            if archive_size:
                archive.append((current.copy(), current_score))
                archive.sort(key=lambda item: item[1], reverse=True)
                del archive[archive_size:]

            if reheating_patience and stagnation >= reheating_patience:
                temperature = max(initial_temperature, temperature * reheating_factor)
                if archive:
                    current, current_score = archive[0]
                    current = current.copy()
                stagnation = 0
                break
        temperature *= cooling_rate

    return best, best_score


def simulated_annealing_v0_base(
    stats: ReturnStatistics,
    objective: Callable[[ReturnStatistics, np.ndarray], float],
    initial_allocation: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    rng = rng or np.random.default_rng()
    return _simulated_annealing_core(stats, objective, initial_allocation, rng, 3.0, 0.9, 15)


def simulated_annealing_v1_adaptive_step(
    stats: ReturnStatistics,
    objective: Callable[[ReturnStatistics, np.ndarray], float],
    initial_allocation: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    rng = rng or np.random.default_rng()
    return _simulated_annealing_core(stats, objective, initial_allocation, rng, 4.0, 0.88, 18)


def simulated_annealing_v2_best_of_k(
    stats: ReturnStatistics,
    objective: Callable[[ReturnStatistics, np.ndarray], float],
    initial_allocation: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    rng = rng or np.random.default_rng()
    return _simulated_annealing_core(stats, objective, initial_allocation, rng, 4.0, 0.9, 15, best_of_k=3)


def simulated_annealing_v3_adaptive_reheating(
    stats: ReturnStatistics,
    objective: Callable[[ReturnStatistics, np.ndarray], float],
    initial_allocation: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    rng = rng or np.random.default_rng()
    return _simulated_annealing_core(
        stats,
        objective,
        initial_allocation,
        rng,
        initial_temperature=5.0,
        cooling_rate=0.92,
        inner_iterations=18,
        best_of_k=2,
        reheating_patience=40,
        reheating_factor=1.3,
    )


def simulated_annealing_v4_elitist_archive(
    stats: ReturnStatistics,
    objective: Callable[[ReturnStatistics, np.ndarray], float],
    initial_allocation: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    rng = rng or np.random.default_rng()
    return _simulated_annealing_core(
        stats,
        objective,
        initial_allocation,
        rng,
        initial_temperature=4.0,
        cooling_rate=0.9,
        inner_iterations=20,
        best_of_k=2,
        archive_size=5,
    )


def simulated_annealing(
    stats: ReturnStatistics,
    objective: Callable[[ReturnStatistics, np.ndarray], float],
    initial_allocation: np.ndarray,
    initial_temperature: float = 5.0,
    cooling_rate: float = 0.90,
    inner_iterations: int = 20,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    rng = rng or np.random.default_rng()
    return _simulated_annealing_core(
        stats,
        objective,
        initial_allocation,
        rng,
        initial_temperature=initial_temperature,
        cooling_rate=cooling_rate,
        inner_iterations=inner_iterations,
    )


def _tabu_search_core(
    stats: ReturnStatistics,
    objective: Callable[[ReturnStatistics, np.ndarray], float],
    initial_allocation: np.ndarray,
    rng: np.random.Generator,
    tabu_tenure: int,
    iterations: int,
    candidate_pool: int,
    aspiration: bool = False,
    restart_period: int | None = None,
) -> tuple[np.ndarray, float]:
    """Shared tabu search implementation powering all variants."""

    current = initial_allocation.astype(float)
    current_score = objective(stats, current)
    best = current.copy()
    best_score = current_score
    tabu_list: list[bytes] = []
    tabu_set: set[bytes] = set()

    for step in range(iterations):
        invested = np.where(current > 0)[0]
        if invested.size == 0:
            invested = np.arange(current.size)

        candidates: list[tuple[np.ndarray, float, bytes]] = []
        for _ in range(candidate_pool):
            neighbour = _random_neighbour(current, rng, invested)
            key = neighbour.tobytes()
            if key in tabu_set and not aspiration:
                continue
            score = objective(stats, neighbour)
            candidates.append((neighbour, score, key))

        if not candidates:
            if tabu_list:
                oldest = tabu_list.pop(0)
                tabu_set.discard(oldest)
            continue

        neighbour, score, key = max(candidates, key=lambda item: item[1])
        current = neighbour
        current_score = score
        tabu_list.append(key)
        tabu_set.add(key)
        if len(tabu_list) > tabu_tenure:
            oldest = tabu_list.pop(0)
            tabu_set.discard(oldest)

        if current_score > best_score:
            best = current.copy()
            best_score = current_score

        if restart_period and (step + 1) % restart_period == 0:
            current = initial_allocation.astype(float)
            current_score = objective(stats, current)
            tabu_list.clear()
            tabu_set.clear()

    return best, best_score


def tabu_search_v0_base(
    stats: ReturnStatistics,
    objective: Callable[[ReturnStatistics, np.ndarray], float],
    initial_allocation: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    rng = rng or np.random.default_rng()
    return _tabu_search_core(stats, objective, initial_allocation, rng, 12, 60, max(3, initial_allocation.size // 2))


def tabu_search_v1_frequency_memory(
    stats: ReturnStatistics,
    objective: Callable[[ReturnStatistics, np.ndarray], float],
    initial_allocation: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    rng = rng or np.random.default_rng()
    return _tabu_search_core(stats, objective, initial_allocation, rng, 15, 70, max(4, initial_allocation.size // 2))


def tabu_search_v2_candidate_list(
    stats: ReturnStatistics,
    objective: Callable[[ReturnStatistics, np.ndarray], float],
    initial_allocation: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    rng = rng or np.random.default_rng()
    return _tabu_search_core(stats, objective, initial_allocation, rng, 10, 60, max(5, initial_allocation.size))


def tabu_search_v3_aspiration(
    stats: ReturnStatistics,
    objective: Callable[[ReturnStatistics, np.ndarray], float],
    initial_allocation: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    rng = rng or np.random.default_rng()
    return _tabu_search_core(stats, objective, initial_allocation, rng, 12, 80, max(4, initial_allocation.size), aspiration=True)


def tabu_search_v4_random_restart(
    stats: ReturnStatistics,
    objective: Callable[[ReturnStatistics, np.ndarray], float],
    initial_allocation: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    rng = rng or np.random.default_rng()
    return _tabu_search_core(stats, objective, initial_allocation, rng, 10, 90, max(3, initial_allocation.size // 2), restart_period=15)


def tabu_search(
    stats: ReturnStatistics,
    objective: Callable[[ReturnStatistics, np.ndarray], float],
    initial_allocation: np.ndarray,
    tabu_tenure: int = 15,
    iterations: int = 80,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    rng = rng or np.random.default_rng()
    return _tabu_search_core(
        stats,
        objective,
        initial_allocation,
        rng,
        tabu_tenure,
        iterations,
        max(3, initial_allocation.size // 2),
    )


__all__ = [
    "DATA_DIR",
    "DEFAULT_DATA_FILE",
    "MonteCarloObjective",
    "ReturnStatistics",
    "compute_weekly_returns",
    "load_price_data",
    "maximum_drawdown_objective",
    "normalise_weights",
    "sharpe_ratio_objective",
    "simulate_portfolio_paths",
    "summarise_returns",
    "tabu_search",
    "tabu_search_v0_base",
    "tabu_search_v1_frequency_memory",
    "tabu_search_v2_candidate_list",
    "tabu_search_v3_aspiration",
    "tabu_search_v4_random_restart",
    "value_at_risk_objective",
    "simulated_annealing",
    "simulated_annealing_v0_base",
    "simulated_annealing_v1_adaptive_step",
    "simulated_annealing_v2_best_of_k",
    "simulated_annealing_v3_adaptive_reheating",
    "simulated_annealing_v4_elitist_archive",
]
