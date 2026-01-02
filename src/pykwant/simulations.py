"""
Simulations Module
==================

This module provides Monte Carlo simulation algorithms for pricing path-dependent
options and analyzing stochastic processes.

It follows a functional design using standard Python lists.
**Note**: Pure Python simulations are slower than NumPy/C++ implementations.
This module is optimized for clarity and educational value rather than raw speed.

Key Functions:
- `generate_paths_gbm`: Simulates Geometric Brownian Motion paths.
- `monte_carlo_price`: Prices any instrument given paths and a payoff function.
"""

import math
import random
from typing import Callable, List

# Type alias for a single price path (time series)
Path = List[float]
# Type alias for a Payoff function: takes a Path, returns a float value
PayoffFn = Callable[[Path], float]


def generate_paths_gbm(
    s0: float,
    drift: float,
    volatility: float,
    time_horizon: float,
    steps: int,
    num_paths: int,
    seed: int | None = None,
) -> List[Path]:
    r"""
    Generates asset price paths using Geometric Brownian Motion (GBM).

    The process follows the SDE:
    $$ dS_t = \\mu S_t dt + \\sigma S_t dW_t $$

    Discretized as:
    $$ S_{t+1} = S_t \\exp\\left( (\\mu - 0.5\\sigma^2)dt + \\sigma\\sqrt{dt} Z \\right) $$

    Args:
        s0 (float): Initial stock price.
        drift (float): Drift rate ($\mu$), usually the risk-free rate $r$ for pricing.
        volatility (float): Annualized volatility ($\sigma$).
        time_horizon (float): Total time in years ($T$).
        steps (int): Number of time steps per path.
        num_paths (int): Number of independent paths to simulate.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        List[Path]: A list of paths, where each path is a list of prices
        starting with s0 and having `steps + 1` elements.
    """
    if seed is not None:
        random.seed(seed)

    dt = time_horizon / steps
    sqrt_dt = math.sqrt(dt)

    # Pre-calculate drift term
    # drift_term = (mu - 0.5 * sigma^2) * dt
    drift_term = (drift - 0.5 * volatility**2) * dt
    vol_term = volatility * sqrt_dt

    all_paths = []

    for _ in range(num_paths):
        path = [s0]
        current_price = s0

        for _ in range(steps):
            # Generate Standard Normal random variable Z
            z = random.gauss(0.0, 1.0)

            # Calculate next price
            # S_next = S_prev * exp(drift_term + vol_term * z)
            growth_factor = math.exp(drift_term + vol_term * z)
            current_price *= growth_factor

            path.append(current_price)

        all_paths.append(path)

    return all_paths


def monte_carlo_price(paths: List[Path], payoff_fn: PayoffFn, discount_factor: float) -> float:
    """
    Calculates the Monte Carlo price of a derivative.

    It applies the `payoff_fn` to each simulated path, takes the average,
    and discounts it to present value.

    Args:
        paths (List[Path]): List of simulated price paths.
        payoff_fn (PayoffFn): A function `f(Path) -> float` defining the instrument payoff.
        discount_factor (float): The discount factor $e^{-rT}$.

    Returns:
        float: The estimated Present Value.
    """
    if not paths:
        return 0.0

    total_payoff = 0.0
    n = len(paths)

    for path in paths:
        total_payoff += payoff_fn(path)

    average_payoff = total_payoff / n
    return average_payoff * discount_factor


# --- Common Payoff Factories ---


def payoff_european_call(strike: float) -> PayoffFn:
    """Returns a payoff function for a European Call: max(S_T - K, 0)."""

    def _payoff(path: Path) -> float:
        return max(path[-1] - strike, 0.0)

    return _payoff


def payoff_european_put(strike: float) -> PayoffFn:
    """Returns a payoff function for a European Put: max(K - S_T, 0)."""

    def _payoff(path: Path) -> float:
        return max(strike - path[-1], 0.0)

    return _payoff


def payoff_asian_arithmetic_call(strike: float) -> PayoffFn:
    """
    Returns a payoff function for an Asian Arithmetic Call.
    Payoff = max(Mean(S) - K, 0).
    """

    def _payoff(path: Path) -> float:
        # Average of the entire path (usually excluding S0, but convention varies)
        # Here we include all points for simplicity.
        avg_price = sum(path) / len(path)
        return max(avg_price - strike, 0.0)

    return _payoff
