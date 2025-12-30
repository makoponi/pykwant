"""
Analytics Module
================

This module provides functions for analyzing financial time series performance.
It focuses on calculating returns, volatility, and risk-adjusted performance metrics
like Sharpe Ratio and Maximum Drawdown.

It follows the functional paradigm: inputs are lists of floats (prices or returns),
and outputs are scalar metrics or transformed lists.
"""

import math

from pykwant import math_utils


def simple_returns(prices: list[float]) -> list[float]:
    r"""
    Calculates the discrete (simple) returns from a series of prices.

    $$ R_t = \frac{P_t - P_{t-1}}{P_{t-1}} $$

    Args:
        prices (list[float]): A list of asset prices ordered by time.

    Returns:
        list[float]: A list of returns. Length is len(prices) - 1.
    """
    if len(prices) < 2:
        return []

    return [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))]


def log_returns(prices: list[float]) -> list[float]:
    r"""
    Calculates the logarithmic (continuously compounded) returns.

    $$ r_t = \ln\left(\frac{P_t}{P_{t-1}}\right) $$

    Args:
        prices (list[float]): A list of asset prices.

    Returns:
        list[float]: A list of log returns. Length is len(prices) - 1.
    """
    if len(prices) < 2:
        return []

    return [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]


def max_drawdown(prices: list[float]) -> float:
    """
    Calculates the Maximum Drawdown (MDD) of a price series.

    MDD is the maximum observed loss from a peak to a trough of a portfolio,
    before a new peak is attained.

    Args:
        prices (list[float]): A list of asset prices.

    Returns:
        float: The maximum drawdown as a positive decimal (e.g., 0.20 for 20% loss).
               Returns 0.0 if prices never drop.
    """
    if not prices:
        return 0.0

    peak = prices[0]
    max_dd = 0.0

    for p in prices:
        if p > peak:
            peak = p
        else:
            dd = (peak - p) / peak
            if dd > max_dd:
                max_dd = dd

    return max_dd


def annualized_volatility(returns: list[float], periods_per_year: int = 252) -> float:
    r"""
    Calculates the Annualized Volatility.

    $$ \sigma_{ann} = \sigma_{period} \times \sqrt{N} $$

    Args:
        returns (list[float]): List of periodic returns.
        periods_per_year (int, optional): Number of periods in a year (252 for daily).
                                          Defaults to 252.

    Returns:
        float: Annualized volatility.
    """
    if len(returns) < 2:
        return 0.0

    std_p = math_utils.std_dev(returns, is_sample=True)
    return std_p * math.sqrt(periods_per_year)


def sharpe_ratio(
    returns: list[float], risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    r"""
    Calculates the Annualized Sharpe Ratio.

    $$ Sharpe = \frac{R_p - R_f}{\sigma_p} $$

    Where $R_p$ is the annualized portfolio return and $\sigma_p$ is the annualized volatility.
    Note: This implementation assumes `risk_free_rate` is the annual risk-free rate.

    Args:
        returns (list[float]): List of periodic returns.
        risk_free_rate (float, optional): Annual risk-free rate (e.g. 0.02). Defaults to 0.0.
        periods_per_year (int, optional): Scaling factor. Defaults to 252.

    Returns:
        float: The Sharpe Ratio.
    """
    if not returns:
        return 0.0

    # 1. Calculate Average Periodic Return
    avg_ret = math_utils.mean(returns)

    # 2. Convert Annual Rf to Periodic Rf (Approximation)
    rf_periodic = risk_free_rate / periods_per_year

    # 3. Excess Return (Periodic)
    excess_ret = avg_ret - rf_periodic

    # 4. Standard Deviation (Periodic)
    std = math_utils.std_dev(returns, is_sample=True)

    if std == 0:
        return 0.0

    # 5. Annualize Sharpe
    # Sharpe_ann = (Excess_periodic / Std_periodic) * sqrt(N)
    return (excess_ret / std) * math.sqrt(periods_per_year)


def sortino_ratio(
    returns: list[float], target_return: float = 0.0, periods_per_year: int = 252
) -> float:
    r"""
    Calculates the Annualized Sortino Ratio.

    Similar to Sharpe Ratio, but penalizes only downside volatility (returns below target).

    $$ Sortino = \frac{R_p - T}{DR} $$

    Where $DR$ is the Downside Risk (Standard deviation of negative asset returns).

    Args:
        returns (list[float]): List of periodic returns.
        target_return (float, optional): Annual target return (MAR). Defaults to 0.0.
        periods_per_year (int, optional): Scaling factor. Defaults to 252.

    Returns:
        float: The Sortino Ratio.
    """
    if not returns:
        return 0.0

    avg_ret = math_utils.mean(returns)
    target_periodic = target_return / periods_per_year

    # Calculate Downside Deviation
    # Sum of (min(0, r - target))^2
    downside_sq_sum = sum(min(0.0, r - target_periodic) ** 2 for r in returns)

    # Sample Downside Deviation
    n = len(returns)
    if n < 2:
        return 0.0

    downside_dev = math.sqrt(downside_sq_sum / (n - 1))

    if downside_dev == 0:
        return 0.0

    excess_ret = avg_ret - target_periodic
    return (excess_ret / downside_dev) * math.sqrt(periods_per_year)
