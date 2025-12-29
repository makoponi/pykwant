r"""
Risk Module
===========

This module calculates financial risk sensitivities (The Greeks) using numerical methods.

Instead of implementing analytical duration formulas for every instrument type,
this module treats the pricing logic as a function $P(r)$ and applies
numerical differentiation to find sensitivities like Duration ($\partial P / \partial r$)
and Convexity ($\partial^2 P / \partial r^2$).

Key Features:
- **Generic**: Works for any instrument that can be priced.
- **Consistent**: Uses the same `numerics` engine as the rest of the library.
- **Functional**: Returns risk metrics as pure values without modifying instruments.
"""

import math
from datetime import date
from typing import Dict

from pykwant import dates, instruments, numerics, rates


def _shift_curve(
    curve: rates.YieldCurveFn, shift: float, reference_date: date
) -> rates.YieldCurveFn:
    """
    Creates a new Yield Curve function with a parallel shift applied.

    This is a higher-order function helper. It wraps the original curve
    and adds a constant spread to the continuously compounded rate.

    Args:
        curve (YieldCurveFn): The original base curve.
        shift (float): The parallel shift size (e.g., 0.0001 for 1bp).
        reference_date (date): The anchor date for time calculation.

    Returns:
        YieldCurveFn: A new closure `f(date) -> df` representing the shifted curve.
    """

    def _shifted_curve(d: date) -> float:
        original_df = curve(d)
        if d == reference_date:
            return original_df

        # We standardise the shift application using ACT/365
        t = dates.act_365(reference_date, d)

        # New DF = Old DF * exp(-shift * t)
        # This corresponds to r_new = r_old + shift
        return original_df * math.exp(-shift * t)

    return _shifted_curve


def pv01(
    instrument: instruments.Instrument,
    curve: rates.YieldCurveFn,
    valuation_date: date,
    bump: float = 1e-4,
) -> float:
    """
    Calculates the PV01 (Present Value of an 01).

    PV01 is the absolute change in the instrument's price for a 1 basis point
    (0.01%) parallel increase in the yield curve.

    $$ PV01 \approx P(r + 1bp) - P(r) $$

    Args:
        instrument (Instrument): The financial instrument to analyze.
        curve (YieldCurveFn): The base market yield curve.
        valuation_date (date): The date of valuation.
        bump (float, optional): The shift size. Defaults to 1e-4 (1 basis point).

    Returns:
        float: The change in value (usually negative for long bond positions).
    """
    # Price with base curve
    p_base = instruments.price_instrument(instrument, curve, valuation_date)

    # Price with shifted curve
    shifted_curve = _shift_curve(curve, bump, valuation_date)
    p_bumped = instruments.price_instrument(instrument, shifted_curve, valuation_date)

    return p_bumped - p_base


def effective_duration(
    instrument: instruments.Instrument, curve: rates.YieldCurveFn, valuation_date: date
) -> float:
    """
    Calculates the Effective Duration.

    Effective Duration measures the percentage sensitivity of the price to
    changes in yield. It is calculated numerically to handle instruments with
    complex cash flows (like options or amortizing bonds).

    $$ Duration = - \frac{1}{P} \frac{dP}{dy} $$

    Args:
        instrument (Instrument): The financial instrument.
        curve (YieldCurveFn): The base yield curve.
        valuation_date (date): The valuation date.

    Returns:
        float: The duration in years (e.g., 4.5). Returns 0.0 if price is 0.
    """

    # Define Price as a function of the shift 's'
    def price_vs_shift(s: float) -> float:
        shifted = _shift_curve(curve, s, valuation_date)
        return float(instruments.price_instrument(instrument, shifted, valuation_date))

    current_price = price_vs_shift(0.0)
    if current_price == 0:
        return 0.0

    # Calculate first derivative dP/dy at s=0
    # We use a slightly larger step (h=1e-4) for stability in financial contexts
    deriv_fn = numerics.numerical_derivative(price_vs_shift, h=1e-4)
    dp_dy = deriv_fn(0.0)

    return -dp_dy / current_price


def effective_convexity(
    instrument: instruments.Instrument, curve: rates.YieldCurveFn, valuation_date: date
) -> float:
    """
    Calculates the Effective Convexity.

    Convexity measures the curvature of the price-yield relationship (the
    second derivative).

    $$ Convexity = \frac{1}{P} \frac{d^2P}{dy^2} $$

    Args:
        instrument (Instrument): The financial instrument.
        curve (YieldCurveFn): The base yield curve.
        valuation_date (date): The valuation date.

    Returns:
        float: The convexity value.
    """

    # Define Price as a function of the shift 's'
    def price_vs_shift(s: float) -> float:
        shifted = _shift_curve(curve, s, valuation_date)
        return float(instruments.price_instrument(instrument, shifted, valuation_date))

    current_price = price_vs_shift(0.0)
    if current_price == 0:
        return 0.0

    # Calculate second derivative d2P/dy2 at s=0
    first_deriv = numerics.numerical_derivative(price_vs_shift, h=1e-4)
    second_deriv = numerics.numerical_derivative(first_deriv, h=1e-4)

    d2p_dy2 = second_deriv(0.0)

    return d2p_dy2 / current_price


def calculate_risk_metrics(
    instrument: instruments.Instrument,
    curve: rates.YieldCurveFn,
    valuation_date: date,
    bump: float = 1e-4,
) -> Dict[str, float]:
    """
    Computes a comprehensive risk report for a single instrument.

    This utility function aggregates Price, Duration, Convexity, and DV01
    into a single dictionary, avoiding redundant recalculations where possible.

    Args:
        instruments (Instrument): The instrument to analyze.
        curve (YieldCurveFn): The yield curve.
        valuation_date (date): The valuation date.
        bump (float, optional): The bump size used for PV01. Defaults to 1bp.

    Returns:
        Dict[str, float]: A dictionary containing keys:
            'price', 'duration', 'convexity', 'dv01'.
    """
    # Note: For maximum efficiency, we could reuse the same pricing calls
    # for duration and convexity (e.g. calculating P_up and P_down once).
    # Here we favor code clarity and reuse of the specific functions.

    price = instruments.price_instrument(instrument, curve, valuation_date)
    dur = effective_duration(instrument, curve, valuation_date)
    conv = effective_convexity(instrument, curve, valuation_date)
    dv01_val = pv01(instrument, curve, valuation_date, bump)

    return {"price": price, "duration": dur, "convexity": conv, "dv01": dv01_val}
