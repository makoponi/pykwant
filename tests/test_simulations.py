"""
Test suite for pykwant.simulations module.
"""

import math
from datetime import date

import pytest

from pykwant import dates, equity, instruments, simulations

# --- Fixtures ---


@pytest.fixture
def flat_curve_5pct():
    """Flat 5% continuously compounded curve."""
    ref_date = date(2025, 1, 1)

    def _curve(d: date) -> float:
        t = dates.act_365(ref_date, d)
        return math.exp(-0.05 * t)

    return _curve


# --- Statistical Properties Tests ---


def test_gbm_martingale_property():
    """
    Test that E[S_T] = S0 * exp(rT) approximately.
    """
    s0 = 100.0
    r = 0.05
    sigma = 0.20
    T = 1.0

    # Generate many paths to converge
    paths = simulations.generate_paths_gbm(
        s0=s0, drift=r, volatility=sigma, time_horizon=T, steps=10, num_paths=5000, seed=42
    )

    # Calculate mean terminal price
    terminal_prices = [p[-1] for p in paths]
    mean_st = sum(terminal_prices) / len(terminal_prices)

    expected_st = s0 * math.exp(r * T)

    # Allow 1% error margin for MC noise
    assert math.isclose(mean_st, expected_st, rel_tol=0.01)


# --- Pricing Tests ---


def test_monte_carlo_european_call_vs_bs(flat_curve_5pct):
    """
    Compare MC pricing of European Call vs Black-Scholes exact formula.
    """
    val_date = date(2025, 1, 1)
    expiry = date(2026, 1, 1)
    spot = 100.0
    strike = 100.0
    vol = 0.20
    r = 0.05
    T = 1.0

    # 1. Black-Scholes Price
    opt = instruments.EuropeanOption("TEST", strike, expiry, "call")
    bs_price = equity.black_scholes_price(opt, spot, vol, flat_curve_5pct, val_date)

    # 2. Monte Carlo Price
    # Generate paths
    paths = simulations.generate_paths_gbm(
        s0=spot, drift=r, volatility=vol, time_horizon=T, steps=50, num_paths=5000, seed=123
    )

    # Define Payoff
    payoff = simulations.payoff_european_call(strike)

    # Discount Factor
    df = flat_curve_5pct(expiry)

    mc_price = simulations.monte_carlo_price(paths, payoff, df)

    # Should be close (e.g. within 2% or 0.20 absolute)
    assert math.isclose(mc_price, bs_price, abs_tol=0.25)


def test_monte_carlo_asian_call():
    """
    Price an Asian Option (Arithmetic Average).
    Asian Call is generally cheaper than European Call because averaging reduces volatility.
    """
    s0 = 100.0
    strike = 100.0
    r = 0.05
    sigma = 0.20
    T = 1.0
    df = math.exp(-r * T)

    paths = simulations.generate_paths_gbm(
        s0=s0, drift=r, volatility=sigma, time_horizon=T, steps=50, num_paths=2000, seed=99
    )

    # European Payoff
    payoff_eu = simulations.payoff_european_call(strike)
    price_eu = simulations.monte_carlo_price(paths, payoff_eu, df)

    # Asian Payoff
    payoff_asian = simulations.payoff_asian_arithmetic_call(strike)
    price_asian = simulations.monte_carlo_price(paths, payoff_asian, df)

    print(f"Euro: {price_eu}, Asian: {price_asian}")

    # Check: Asian < European for ATM options usually
    assert price_asian < price_eu
    # Check: Positive price
    assert price_asian > 0.0


def test_empty_paths():
    assert simulations.monte_carlo_price([], lambda p: 0.0, 1.0) == 0.0
