"""
Portfolio Module
================

This module handles the management and risk aggregation of portfolios.

In PyKwant, a **Portfolio** is not a class, but simply a list of `Position` objects.
This module provides pure functions to aggregate these positions, calculating
total Net Present Value (NPV), risk metrics (Portfolio Duration, DV01), and
exposure reports.

Key Concepts:
- **Position**: A tuple of (Instrument, Quantity).
- **Portfolio**: A `list[Position]`.
"""

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, TypeAlias

from pykwant import instruments, rates, risk


@dataclass(frozen=True)
class Position:
    """
    Immutable representation of a holding in a financial instrument.

    Attributes:
        instrument (instruments.Instrument): The financial instrument held.
        quantity (float): The amount held.
                          Positive (+) for Long positions.
                          Negative (-) for Short positions.
    """

    instrument: instruments.Instrument
    quantity: float


# Type Alias for a Portfolio (just a list of positions)
Portfolio: TypeAlias = List[Position]


def portfolio_npv(portfolio: Portfolio, curve: rates.YieldCurveFn, valuation_date: date) -> float:
    r"""
    Calculates the Total Net Present Value (NPV) of a portfolio.

    It sums the market value of each position:
    $$ Total NPV = \sum (Price_i \times Quantity_i) $$

    Args:
        portfolio (Portfolio): A list of positions.
        curve (YieldCurveFn): The market yield curve for pricing.
        valuation_date (date): The date of valuation.

    Returns:
        float: The total market value of the portfolio.
    """
    total_value = 0.0
    for pos in portfolio:
        price = instruments.price_instrument(pos.instrument, curve, valuation_date)
        total_value += price * pos.quantity
    return total_value


def portfolio_risk(
    portfolio: Portfolio, curve: rates.YieldCurveFn, valuation_date: date
) -> Dict[str, float]:
    r"""
    Aggregates risk metrics for the entire portfolio.

    Calculates:
    - **Market Value**: Total NPV.
    - **Total DV01**: The dollar change of the portfolio for a 1bp shift.
      (Sum of individual DV01 * Quantity).
    - **Portfolio Duration**: The value-weighted average duration.
      $$ Dur_{port} = \frac{\sum (Dur_i \times Value_i)}{Total Value} $$

    Args:
        portfolio (Portfolio): A list of positions.
        curve (YieldCurveFn): The market yield curve.
        valuation_date (date): The date of valuation.

    Returns:
        Dict[str, float]: A dictionary containing 'market_value', 'total_dv01',
        and 'portfolio_duration'.
    """
    total_value = 0.0
    total_dv01 = 0.0
    weighted_duration_sum = 0.0

    for pos in portfolio:
        # Calculate metrics for the single instrument
        metrics = risk.calculate_risk_metrics(pos.instrument, curve, valuation_date)

        position_value = metrics["price"] * pos.quantity
        position_dv01 = metrics["dv01"] * pos.quantity

        # Aggregate
        total_value += position_value
        total_dv01 += position_dv01

        # For Portfolio Duration: Sum(Duration * MarketValue)
        weighted_duration_sum += metrics["duration"] * position_value

    # Handle edge case where portfolio value is zero (to avoid division by zero)
    if total_value == 0:
        port_duration = 0.0
    else:
        port_duration = weighted_duration_sum / total_value

    return {
        "market_value": total_value,
        "total_dv01": total_dv01,
        "portfolio_duration": port_duration,
    }


def exposure_by_maturity_year(
    portfolio: Portfolio, curve: rates.YieldCurveFn, valuation_date: date
) -> Dict[int, float]:
    """
    Aggregates portfolio exposure (NPV) grouped by maturity year.

    Useful for analyzing the maturity profile (bucketing) of the portfolio.

    Args:
        portfolio (Portfolio): A list of positions.
        curve (YieldCurveFn): The market yield curve.
        valuation_date (date): The date of valuation.

    Returns:
        Dict[int, float]: A dictionary mapping Year -> Total NPV for that year.
        Example: {2025: 1500.0, 2026: -500.0}
    """
    exposure: Dict[int, float] = {}

    for pos in portfolio:
        # Determine maturity year based on instrument type
        maturity_year = 0

        if hasattr(pos.instrument, "maturity_date"):
            # Works for FixedRateBond and similar
            maturity_year = pos.instrument.maturity_date.year
        elif hasattr(pos.instrument, "expiry_date"):
            # Works for Options
            maturity_year = pos.instrument.expiry_date.year
        else:
            # Fallback for instruments without clear maturity (e.g. Cash)
            maturity_year = valuation_date.year

        # Calculate Value
        price = instruments.price_instrument(pos.instrument, curve, valuation_date)
        value = price * pos.quantity

        # Accumulate in bucket
        exposure[maturity_year] = exposure.get(maturity_year, 0.0) + value

    return exposure
