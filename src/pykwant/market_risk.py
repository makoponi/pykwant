"""
Market Risk Module
==================

This module provides functions to calculate Market Risk metrics such as
Value at Risk (VaR) and Expected Shortfall (CVaR/ES).

It supports both:
1. **Parametric Approach** (Variance-Covariance): Assumes normal distribution.
2. **Historical Approach**: Uses historical simulation data.

All calculations are performed using pure Python and the `math_utils` module.
"""

import math

from pykwant import math_utils


def parametric_var(
    portfolio_value: float,
    volatility: float,
    confidence_level: float = 0.95,
    horizon_days: int = 1,
    periods_per_year: int = 252,
) -> float:
    r"""
    Calculates the Parametric Value at Risk (VaR) assuming a Normal Distribution.

    $$ VaR = V \cdot \sigma_{horizon} \cdot z_{\\alpha} $$

    Where:
    - $V$ is the portfolio value.
    - $\\sigma_{horizon} = \\sigma_{annual} / \\sqrt{252} \\cdot \\sqrt{t}$.
    - $z_{\\alpha}$ is the inverse cumulative normal distribution at $\\alpha$.

    Args:
        portfolio_value (float): Current market value of the portfolio.
        volatility (float): Annualized volatility (sigma) e.g., 0.20 for 20%.
        confidence_level (float, optional): Confidence level (e.g., 0.95 or 0.99). Defaults to 0.95.
        horizon_days (int, optional): Time horizon in days. Defaults to 1.
        periods_per_year (int, optional): Trading days in a year. Defaults to 252.

    Returns:
        float: The estimated maximum loss (positive number).
    """
    if not (0 < confidence_level < 1):
        raise ValueError("Confidence level must be between 0 and 1.")

    # Calculate alpha (tail probability)
    # For 95% confidence, we look at the bottom 5% tail.
    alpha = 1.0 - confidence_level

    # Get the z-score (number of std devs) for the tail
    # norm_ppf(0.05) ~ -1.645. We want the absolute distance.
    z_score = abs(math_utils.norm_ppf(alpha))

    # Scale volatility to the horizon
    # sigma_daily = sigma_annual / sqrt(252)
    # sigma_horizon = sigma_daily * sqrt(horizon)
    sigma_horizon = volatility * math.sqrt(horizon_days / periods_per_year)

    return portfolio_value * sigma_horizon * z_score


def historical_var(
    portfolio_value: float, returns: list[float], confidence_level: float = 0.95
) -> float:
    r"""
    Calculates the Historical Value at Risk (VaR) using historical simulation.

    It finds the percentile of the historical returns distribution corresponding
    to the specified confidence level.

    Args:
        portfolio_value (float): Current market value of the portfolio.
        returns (list[float]): List of historical period returns (e.g., daily).
        confidence_level (float, optional): Confidence level (e.g., 0.95). Defaults to 0.95.

    Returns:
        float: The estimated maximum loss (positive number).
    """
    if not returns:
        raise ValueError("Returns list cannot be empty.")

    alpha = 1.0 - confidence_level

    # Calculate the percentile of returns (e.g., 5th percentile for 95% confidence)
    # We expect returns to be mostly negative in the tail.
    worst_return = math_utils.percentile(returns, alpha)

    # VaR is the loss amount. If return is -0.05, VaR is 0.05 * Value.
    # We return a positive number representing the loss.
    return portfolio_value * abs(worst_return)


def parametric_expected_shortfall(
    portfolio_value: float,
    volatility: float,
    confidence_level: float = 0.95,
    horizon_days: int = 1,
    periods_per_year: int = 252,
) -> float:
    r"""
    Calculates the Parametric Expected Shortfall (ES), also known as Conditional VaR (CVaR).

    ES is the average loss given that the loss exceeds the VaR.
    For a Normal Distribution:
    $$ ES = V \cdot \\sigma_{horizon} \cdot \\frac{\\phi(z_{\\alpha})}{1 - \\alpha} $$

    Args:
        portfolio_value (float): Current market value.
        volatility (float): Annualized volatility.
        confidence_level (float, optional): Confidence level (e.g., 0.95). Defaults to 0.95.
        horizon_days (int, optional): Horizon in days. Defaults to 1.
        periods_per_year (int, optional): Periods per year. Defaults to 252.

    Returns:
        float: The expected shortfall amount.
    """
    if not (0 < confidence_level < 1):
        raise ValueError("Confidence level must be between 0 and 1.")

    alpha = 1.0 - confidence_level
    z_score = math_utils.norm_ppf(alpha)  # Negative value e.g. -1.645

    # PDF at the z-score
    pdf_z = math_utils.norm_pdf(z_score)

    # Volatility over horizon
    sigma_horizon = volatility * math.sqrt(horizon_days / periods_per_year)

    # ES Formula factor
    es_factor = pdf_z / alpha

    return portfolio_value * sigma_horizon * es_factor


def historical_expected_shortfall(
    portfolio_value: float, returns: list[float], confidence_level: float = 0.95
) -> float:
    r"""
    Calculates the Historical Expected Shortfall (CVaR).

    It calculates the average of all historical returns that are worse than the
    Historical VaR threshold.

    Args:
        portfolio_value (float): Current market value.
        returns (list[float]): List of historical returns.
        confidence_level (float, optional): Confidence level. Defaults to 0.95.

    Returns:
        float: The expected shortfall amount.
    """
    if not returns:
        raise ValueError("Returns list cannot be empty.")

    alpha = 1.0 - confidence_level
    cutoff = math_utils.percentile(returns, alpha)

    # Filter returns worse than cutoff (tail)
    tail_returns = [r for r in returns if r <= cutoff]

    if not tail_returns:
        # Should not happen given percentile logic unless list is very short or uniform
        return portfolio_value * abs(cutoff)

    avg_tail_return = math_utils.mean(tail_returns)

    return portfolio_value * abs(avg_tail_return)
