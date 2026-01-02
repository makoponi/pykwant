"""
Optimization Module
===================

This module provides tools for Portfolio Optimization using Monte Carlo simulations.

Instead of using complex quadratic programming solvers (which require heavy dependencies),
this module generates thousands of random portfolios to approximate the **Efficient Frontier**.

It identifies:
1. **Max Sharpe Ratio Portfolio**: The tangency portfolio (optimal risk-adjusted return).
2. **Min Volatility Portfolio**: The portfolio with the lowest possible risk.

This approach is "Zero-Dependency" and relies on pure Python math.
"""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

# --- Data Structures ---


@dataclass(frozen=True)
class PortfolioStats:
    """
    Container for optimized portfolio results.
    """

    weights: List[float]
    expected_return: float
    volatility: float
    sharpe_ratio: float


# --- Core Functions ---


def _portfolio_variance(weights: List[float], cov_matrix: List[List[float]]) -> float:
    r"""
    Calculates portfolio variance: w^T * Sigma * w

    $$ \sigma_p^2 = \sum_i \sum_j w_i w_j \sigma_{ij} $$
    """
    variance = 0.0
    n = len(weights)
    for i in range(n):
        for j in range(n):
            variance += weights[i] * weights[j] * cov_matrix[i][j]
    return variance


def _portfolio_return(weights: List[float], expected_returns: List[float]) -> float:
    """
    Calculates expected portfolio return: w^T * mu
    """
    return sum(w * r for w, r in zip(weights, expected_returns, strict=True))


def generate_random_weights(n: int) -> List[float]:
    """
    Generates N random weights that sum to 1.0 (Long Only).
    """
    # Generate random numbers
    raw = [random.random() for _ in range(n)]
    total = sum(raw)

    # Normalize
    return [x / total for x in raw]


def optimize_portfolio_monte_carlo(
    expected_returns: List[float],
    cov_matrix: List[List[float]],
    num_portfolios: int = 10_000,
    risk_free_rate: float = 0.0,
    seed: int | None = None,
) -> Tuple[PortfolioStats, PortfolioStats]:
    """
    Finds the optimal portfolios by simulating random weight allocations.

    It returns two specific portfolios:
    1. **Max Sharpe**: Maximizes (Return - Rf) / Volatility.
    2. **Min Volatility**: Minimizes Volatility (Global Minimum Variance).

    Args:
        expected_returns (List[float]): List of annual expected returns for each asset.
        cov_matrix (List[List[float]]): Covariance matrix (NxN) of returns.
        num_portfolios (int, optional): Number of simulations. Defaults to 10,000.
        risk_free_rate (float, optional): Annual risk-free rate. Defaults to 0.0.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        Tuple[PortfolioStats, PortfolioStats]: (Max Sharpe Portfolio, Min Vol Portfolio).
    """
    if seed is not None:
        random.seed(seed)

    n_assets = len(expected_returns)
    if len(cov_matrix) != n_assets or len(cov_matrix[0]) != n_assets:
        raise ValueError("Covariance matrix dimensions must match number of assets.")

    # Trackers for optimal portfolios
    max_sharpe_stats = None
    max_sharpe_val = -float("inf")

    min_vol_stats = None
    min_vol_val = float("inf")

    for _ in range(num_portfolios):
        weights = generate_random_weights(n_assets)

        # Calculate Stats
        p_ret = _portfolio_return(weights, expected_returns)
        p_var = _portfolio_variance(weights, cov_matrix)
        p_vol = math.sqrt(p_var)

        # Calculate Sharpe (handle zero vol edge case)
        if p_vol > 0:
            p_sharpe = (p_ret - risk_free_rate) / p_vol
        else:
            p_sharpe = 0.0

        current_stats = PortfolioStats(weights, p_ret, p_vol, p_sharpe)

        # Check Max Sharpe
        if p_sharpe > max_sharpe_val:
            max_sharpe_val = p_sharpe
            max_sharpe_stats = current_stats

        # Check Min Volatility
        if p_vol < min_vol_val:
            min_vol_val = p_vol
            min_vol_stats = current_stats

    # Should generally not happen unless num_portfolios is 0
    if max_sharpe_stats is None or min_vol_stats is None:
        raise ValueError("Optimization failed. Ensure num_portfolios > 0.")

    return max_sharpe_stats, min_vol_stats
