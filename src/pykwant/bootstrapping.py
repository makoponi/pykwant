"""
Bootstrapping Module
====================

This module constructs Yield Curves from market quotes (Deposits, Swaps).

It uses a sequential **Bootstrapping** algorithm:
1. Sort instruments by maturity.
2. For each instrument, solve for the Zero Rate that reprices it to Par.
3. Extend the curve point-by-point.

The resulting curve is a standard `YieldCurveFn` compatible with the rest of the library.
"""

import math
from dataclasses import dataclass
from datetime import date
from typing import Callable, List, Union

from pykwant import dates, instruments, numerics, rates

# --- Calibration Helpers ---


@dataclass(frozen=True)
class DepositRate:
    """
    Represents a cash deposit rate (e.g., LIBOR/EURIBOR).
    Simple interest: 1 + r * t = 1 / DF
    """

    rate: float
    maturity_date: date
    day_count: dates.DayCountConvention = dates.act_360


@dataclass(frozen=True)
class SwapRate:
    """
    Represents a Par Swap Rate.
    Modeled as a Fixed Rate Bond trading at Par (100.0).
    """

    rate: float
    maturity_date: date
    frequency_months: int
    day_count: dates.DayCountConvention = dates.thirty_360
    calendar: dates.Calendar = dates.Calendar()  # Default calendar


CalibrationInstrument = Union[DepositRate, SwapRate]


def _price_helper(
    helper: CalibrationInstrument, curve: rates.YieldCurveFn, valuation_date: date
) -> float:
    """Internal dispatcher to price calibration instruments."""

    if isinstance(helper, DepositRate):
        # Value of 1 unit invested at start
        t = helper.day_count(valuation_date, helper.maturity_date)
        future_value = 1.0 + helper.rate * t
        return future_value * curve(helper.maturity_date)

    if isinstance(helper, SwapRate):
        # Value of a Fixed Rate Bond with Coupon = SwapRate
        # If the curve is correct, Price should be 1.0 (Par)
        bond = instruments.FixedRateBond(
            face_value=instruments.Money(1.0),  # Normalized to 1
            coupon_rate=helper.rate,
            start_date=valuation_date,
            maturity_date=helper.maturity_date,
            frequency_months=helper.frequency_months,
            day_count=helper.day_count,
            calendar=helper.calendar,
        )
        return instruments.price_instrument(bond, curve, valuation_date)

    raise NotImplementedError(f"Unknown helper type: {type(helper)}")


def bootstrap_curve(
    valuation_date: date,
    helpers: List[CalibrationInstrument],
    interpolation_method: Callable[
        [List[float], List[float], bool], Callable[[float], float]
    ] = numerics.log_linear_interpolation,
) -> rates.YieldCurveFn:
    """
    Constructs a Yield Curve by bootstrapping market instruments.

    The algorithm sorts instruments by maturity and iteratively solves for the
    discount factor at each pillar that makes the instrument price equal to its target.

    Args:
        valuation_date (date): The anchor date (t=0).
        helpers (List[CalibrationInstrument]): List of Deposits/Swaps.
        interpolation_method: Factory function for interpolation (default: log-linear).

    Returns:
        rates.YieldCurveFn: A callable curve function.
    """
    # 1. Sort helpers by maturity
    sorted_helpers = sorted(helpers, key=lambda h: h.maturity_date)

    # 2. Initialize Curve Pillars with t=0
    known_dates = [valuation_date]
    known_dfs = [1.0]

    # Pre-calculate time fractions for interpolation to speed up
    # We use ACT/365 for the internal interpolation grid
    known_times = [0.0]

    for instrument in sorted_helpers:
        mat_date = instrument.maturity_date

        # Skip if duplicate date (simple handling)
        if mat_date <= known_dates[-1]:
            continue

        mat_time = dates.act_365(valuation_date, mat_date)

        # Define objective function to find Zero Rate at this maturity
        # We solve for 'r' where DF = exp(-r * t)
        def objective(
            r_guess: float,
            mat_time: float = mat_time,
            instrument: CalibrationInstrument = instrument,
        ) -> float:
            df_guess = math.exp(-r_guess * mat_time)

            # Construct temporary lists for the interpolator
            temp_times = known_times + [mat_time]
            temp_dfs = known_dfs + [df_guess]

            # Create temp interpolator
            # Note: We interpolate on times/DFs directly
            interp_fn = interpolation_method(temp_times, temp_dfs, True)

            # Create temp curve function wrapper
            def temp_curve(d: date) -> float:
                t = dates.act_365(valuation_date, d)
                val = interp_fn(t)
                # Handle log-linear logic if the interpolator returns log values?
                # numerics.log_linear_interpolation returns a function that outputs values directly.
                return val

            # Calculate Price
            price = _price_helper(instrument, temp_curve, valuation_date)

            # Target:
            # Deposit: 1.0 (PV of 1 unit invested)
            # Swap: 1.0 (Par)
            return price - 1.0

        # Initial guess: Use previous zero rate or small number
        # If it's the first point, guess rate from instrument if possible, or 0.02
        prev_df = known_dfs[-1]
        prev_time = known_times[-1]

        # Simple guess
        if prev_time > 0:
            guess_r = -math.log(prev_df) / prev_time
        else:
            guess_r = 0.05

        # Solve
        implied_r = numerics.newton_solve(objective, target=0.0, guess=guess_r)

        if implied_r is None:
            raise ValueError(f"Bootstrapping failed at {mat_date}")

        # Store result
        final_df = math.exp(-implied_r * mat_time)
        known_dates.append(mat_date)
        known_dfs.append(final_df)
        known_times.append(mat_time)

    # 3. Create Final Curve
    # Use the rates module factory to ensure consistency
    return rates.create_curve_from_discount_factor(
        reference_date=valuation_date,
        dates_list=known_dates,
        dfs_list=known_dfs,
        day_count=dates.act_365,
    )
