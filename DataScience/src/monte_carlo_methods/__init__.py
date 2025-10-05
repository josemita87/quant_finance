"""Convenience exports for Monte Carlo portfolio helpers."""

from .monte_carlo_portfolio import (
    DATA_DIR,
    DEFAULT_DATA_FILE,
    ReturnStatistics,
    compute_weekly_returns,
    load_price_data,
    maximum_drawdown_objective,
    normalise_weights,
    sharpe_ratio_objective,
    simulate_portfolio_paths,
    summarise_returns,
    tabu_search,
    tabu_search_v0_base,
    tabu_search_v1_frequency_memory,
    tabu_search_v2_candidate_list,
    tabu_search_v3_aspiration,
    tabu_search_v4_random_restart,
    value_at_risk_objective,
    simulated_annealing,
    simulated_annealing_v0_base,
    simulated_annealing_v1_adaptive_step,
    simulated_annealing_v2_best_of_k,
    simulated_annealing_v3_adaptive_reheating,
    simulated_annealing_v4_elitist_archive,
)


def run_cli() -> None:
    """Lazy wrapper around the command-line entry point."""

    from .main import run_cli as _run_cli

    _run_cli()

__all__ = [
    "DATA_DIR",
    "DEFAULT_DATA_FILE",
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
    "run_cli",
]
