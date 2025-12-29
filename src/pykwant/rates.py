"""
Rates Module
============

This module handles interest rate modelling, yield curve construction, and
fundamental rate calculations.

It follows a functional design where a **Yield Curve** is simply a function
mapping a date to a Discount Factor:
    `f(date) -> float`

This allows curves to be composed, interpolated, or mocked easily without
complex inheritance hierarchies.
"""

import math
from datetime import date
from typing import Callable, TypeAlias

from pykwant import dates, numerics

# Type Alias for a Yield Curve: A function that takes a date and returns a Discount Factor.
YieldCurveFn: TypeAlias = Callable[[date], float]


def compound_factor(rate: float, t: float, frequency: int = 0) -> float:
    """
    Calculates the compounding factor for a given rate and time.

    The compounding factor is the growth multiplier for a unit amount.
    - Continuous compounding: $e^{rt}$
    - Discrete compounding: $(1 + r/f)^{ft}$

    Args:
        rate (float): The interest rate (decimal, e.g., 0.05 for 5%).
        t (float): The time fraction in years.
        frequency (int, optional): The compounding frequency per year.
            Use 0 for continuous compounding.
            Use 1 for annual, 2 for semi-annual, etc. Defaults to 0.

    Returns:
        float: The compounding factor (usually > 1 for positive rates).
    """
    if frequency == 0:
        return math.exp(rate * t)
    return float((1 + rate / frequency) ** (frequency * t))


def create_curve_from_discount_factor(
    reference_date: date,
    dates_list: list[date],
    dfs_list: list[float],
    day_count: dates.DayCountConvention = dates.act_365,
) -> YieldCurveFn:
    """
    Creates a Yield Curve function from a set of Discount Factors.

    This factory function returns a closure that interpolates Discount Factors
    log-linearly between the provided pillars.

    Args:
        reference_date (date): The anchor date for the curve (t=0).
        dates_list (list[date]): Sorted list of pillar dates.
        dfs_list (list[float]): Corresponding discount factors for the pillar dates.
        day_count (DayCountConvention, optional): The function to convert dates
            to time fractions. Defaults to dates.act_365.

    Returns:
        YieldCurveFn: A callable `f(date) -> discount_factor`.
    """
    # Calculate time fractions (t) for the pillars relative to reference_date
    times = [day_count(reference_date, d) for d in dates_list]

    # Create a log-linear interpolator: f(t) -> DF
    interpolator = numerics.log_linear_interpolation(times, dfs_list)

    # Define the closure that acts as the Yield Curve
    def _curve(d: date) -> float:
        if d == reference_date:
            return 1.0
        t = day_count(reference_date, d)
        return interpolator(t)

    return _curve


def zero_rates(
    curve: YieldCurveFn,
    reference_date: date,
    target_date: date,
    day_count: dates.DayCountConvention = dates.act_365,
) -> float:
    r"""
    Extracts the Zero (Spot) Rate from a yield curve for a specific date.

    The Zero Rate is the continuously compounded rate $r$ such that:
    $DF(t) = e^{-r t}$ implies $r = -\ln(DF) / t$

    Args:
        curve (YieldCurveFn): The yield curve function.
        reference_date (date): The start date (t=0).
        target_date (date): The date for which the rate is requested.
        day_count (DayCountConvention, optional): Day count for time calculation.
            Defaults to dates.act_365.

    Returns:
        float: The zero rate (decimal). Returns 0.0 if target_date == reference_date.
    """
    if target_date == reference_date:
        return 0.0

    df = curve(target_date)
    t = day_count(reference_date, target_date)

    # Avoid division by zero if t is extremely small but not zero
    if t == 0:
        return 0.0

    return -math.log(df) / t


def forward_rate(
    curve: YieldCurveFn,
    start_date: date,
    end_date: date,
    day_count: dates.DayCountConvention = dates.act_365,
) -> float:
    r"""
    Calculates the implied Forward Rate between two future dates.

    The forward rate is the rate $F$ applicable between $T_1$ and $T_2$.
    Based on the ratio of discount factors:
    $F = \frac{1}{\tau} \left( \frac{DF(T_1)}{DF(T_2)} - 1 \right)$ (Simple Compounding approx.)
    or continuously compounded equivalent depending on convention.
    Here we return the simple forward rate implied by the discount factors.

    Args:
        curve (YieldCurveFn): The yield curve function.
        start_date (date): The start of the forward period.
        end_date (date): The end of the forward period.
        day_count (DayCountConvention, optional): Day count. Defaults to dates.act_365.

    Returns:
        float: The annualized forward rate.
    """
    df_start = curve(start_date)
    df_end = curve(end_date)
    tau = day_count(start_date, end_date)

    if tau == 0:
        return 0.0

    return (df_start / df_end - 1.0) / tau


def present_value(amount: float, payment_date: date, curve: YieldCurveFn) -> float:
    r"""
    Calculates the Present Value (PV) of a single future cash flow.

    Args:
        amount (float): The monetary amount of the cash flow.
        payment_date (date): The date the payment occurs.
        curve (YieldCurveFn): The yield curve used for discounting.

    Returns:
        float: The discounted value ($Amount \times DF(payment\_date)$).
    """
    return amount * curve(payment_date)
