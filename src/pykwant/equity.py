"""
Equity Module
=============

This module provides pricing functions for equity derivatives, specifically
focusing on the Black-Scholes-Merton framework.

It adheres to the library's functional paradigm:
- Inputs: Immutable data structures (EuropeanOption) and market data (Spot, Vol, Curve).
- Output: The calculated price (Money).
"""

import math
from datetime import date

from pykwant import dates, instruments, math_utils, rates


def black_scholes_price(
    option: instruments.EuropeanOption,
    spot: float,
    volatility: float,
    curve: rates.YieldCurveFn,
    valuation_date: date,
) -> instruments.Money:
    """
    Prices a European Option using the Black-Scholes-Merton formula.

    This function calculates the theoretical value of a Vanilla Call or Put option.
    It automatically extracts the implied continuous risk-free rate ($r$) from the
    provided yield curve for the specific time to maturity.

    Formulas:
    $$ d_1 = \\frac{\\ln(S/K) + (r + \\sigma^2/2)T}{\\sigma\\sqrt{T}} $$
    $$ d_2 = d_1 - \\sigma\\sqrt{T} $$
    $$ Call = S \\Phi(d_1) - K e^{-rT} \\Phi(d_2) $$
    $$ Put = K e^{-rT} \\Phi(-d_2) - S \\Phi(-d_1) $$

    Args:
        option (instruments.EuropeanOption): The option instrument definition
            (Strike, Expiry, Call/Put type).
        spot (float): The current spot price of the underlying asset ($S_0$).
        volatility (float): The annualized volatility ($\\sigma$) in decimal
            format (e.g., 0.20 for 20%).
        curve (rates.YieldCurveFn): The risk-free yield curve function used
            to extract the discount factor and implied rate.
        valuation_date (date): The date on which valuation is performed.

    Returns:
        instruments.Money: The calculated option price. Returns 0.0 if the
        valuation date is past the expiry date.
    """
    # 1. Validation
    if valuation_date > option.expiry_date:
        return instruments.Money(0.0)

    # 2. Parameters setup
    # Time to maturity in years (ACT/365 is standard for BSM)
    T = dates.act_365(valuation_date, option.expiry_date)
    K = float(option.strike)
    S = spot
    sigma = volatility

    # Extract Discount Factor and Implied Rate (r) from the curve
    # DF = e^(-rT)  =>  r = -ln(DF) / T
    df = curve(option.expiry_date)

    # Handle extremely short time to avoid division by zero
    if T <= 1e-9:
        intrinsic = max(S - K, 0.0) if option.call_put == "call" else max(K - S, 0.0)
        return instruments.Money(intrinsic)

    r = -math.log(df) / T

    # 3. Handle Edge Case: Zero Volatility
    if sigma == 0:
        # If vol is zero, the option value is the intrinsic value discounted
        # (max(S - K*e^-rT, 0))
        forward_price = S * math.exp(r * T)
        if option.call_put == "call":
            payoff = max(forward_price - K, 0.0)
        else:
            payoff = max(K - forward_price, 0.0)
        return instruments.Money(payoff * df)

    # 4. d1, d2 Calculation
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    # 5. Pricing (using standard normal CDF)
    if option.call_put == "call":
        price = S * math_utils.norm_cdf(d1) - K * df * math_utils.norm_cdf(d2)
    else:
        # Put-Call Parity: P = C - S + K * df
        # Or direct formula:
        price = K * df * math_utils.norm_cdf(-d2) - S * math_utils.norm_cdf(-d1)

    return instruments.Money(max(price, 0.0))
