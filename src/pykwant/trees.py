"""
Trees Module
============

This module provides pricing algorithms based on Lattice methods (Binomial Trees).
It is primarily used for pricing American Options, which can be exercised at any point
before expiration.

The implementation uses the **Cox-Ross-Rubinstein (CRR)** model.
It is designed to be memory efficient by storing only the current time-step layer,
avoiding full matrix allocation.
"""

import math
from datetime import date

from pykwant import dates, instruments, rates


def _payoff(option_type: str, spot: float, strike: float) -> float:
    """Internal helper to calculate intrinsic value."""
    if option_type == "call":
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)


def binomial_price(
    option: instruments.AmericanOption,
    spot: float,
    volatility: float,
    curve: rates.YieldCurveFn,
    valuation_date: date,
    steps: int = 100,
) -> float:
    r"""
    Prices an American Option using a Binomial Tree (CRR Model).

    The algorithm constructs a recombination tree of asset prices and traverses it
    backwards from maturity to valuation date. At each node, it checks for
    early exercise optimality.

    $$ Value = \\max( Intrinsic, Continuation ) $$

    Args:
        option (AmericanOption): The option instrument.
        spot (float): Current spot price of the underlying.
        volatility (float): Annualized volatility (e.g., 0.20).
        curve (YieldCurveFn): Risk-free yield curve.
        valuation_date (date): Date of valuation.
        steps (int, optional): Number of time steps in the tree. Defaults to 100.

    Returns:
        float: The option price.
    """
    if valuation_date > option.expiry_date:
        return 0.0

    # 1. Setup Parameters
    T = dates.act_365(valuation_date, option.expiry_date)
    if T <= 0:
        return _payoff(option.call_put, spot, float(option.strike))

    dt = T / steps

    # Extract continuous rate r
    df_T = curve(option.expiry_date)
    r = -math.log(df_T) / T

    # CRR Parameters
    # u = exp(sigma * sqrt(dt))
    # d = 1 / u
    # p = (exp(r*dt) - d) / (u - d)
    u = math.exp(volatility * math.sqrt(dt))
    d = 1.0 / u
    discount_factor_step = math.exp(-r * dt)

    growth_factor = math.exp(r * dt)
    p = (growth_factor - d) / (u - d)

    # 2. Generate Leaf Nodes (Payoffs at Maturity)
    # At step N, there are N+1 nodes.
    # Prices: S * u^(N-i) * d^i  for i in 0..N
    # We store values in a list.

    values = []
    strike = float(option.strike)

    for i in range(steps + 1):
        # Number of down moves = i
        # Number of up moves = steps - i
        S_T = spot * (u ** (steps - i)) * (d**i)
        values.append(_payoff(option.call_put, S_T, strike))

    # 3. Backward Induction
    # Reduce the list by 1 element at each step
    for step_index in range(steps - 1, -1, -1):
        new_values = []
        for i in range(step_index + 1):
            # Underlying price at this node (step_index, i)
            # Up moves = step_index - i, Down moves = i
            S_node = spot * (u ** (step_index - i)) * (d**i)

            # Continuation Value (Discounted expected future value)
            # values[i] is the "Up" child (from previous loop perspective of i)
            # values[i+1] is the "Down" child
            # Wait, loops:
            # Previous layer (step N): [u^N, u^N-1*d, ..., d^N]
            # Index i in step N corresponds to i down moves.
            # Node (N-1, i) connects to (N, i) [UP] and (N, i+1) [DOWN]

            continuation = discount_factor_step * (p * values[i] + (1 - p) * values[i + 1])

            # Intrinsic Value (Early Exercise)
            intrinsic = _payoff(option.call_put, S_node, strike)

            # American Option Logic
            new_values.append(max(continuation, intrinsic))

        values = new_values

    return values[0]
