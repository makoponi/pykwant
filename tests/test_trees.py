"""
Test suite for pykwant.trees module.
"""

import math
from datetime import date

import pytest

from pykwant import dates, equity, instruments, trees

# --- Fixtures ---


@pytest.fixture
def flat_curve_5pct():
    """Flat 5% continuously compounded curve."""
    ref_date = date(2025, 1, 1)

    def _curve(d: date) -> float:
        t = dates.act_365(ref_date, d)
        return math.exp(-0.05 * t)

    return _curve


@pytest.fixture
def american_put():
    return instruments.AmericanOption(
        asset_name="TEST",
        strike=instruments.Money(100.0),
        expiry_date=date(2026, 1, 1),
        call_put="put",
    )


# --- Tests ---


def test_binomial_price_convergence_european(flat_curve_5pct):
    # An American Call on non-dividend paying stock = European Call.
    # We can check if Tree converges to Black-Scholes.

    val_date = date(2025, 1, 1)
    expiry = date(2026, 1, 1)
    spot = 100.0
    vol = 0.20

    # Define instruments
    am_call = instruments.AmericanOption("TEST", 100.0, expiry, "call")
    eu_call = instruments.EuropeanOption("TEST", 100.0, expiry, "call")

    # 1. Black Scholes Price
    bs_price = equity.black_scholes_price(eu_call, spot, vol, flat_curve_5pct, val_date)

    # 2. Tree Price (High steps for convergence)
    tree_price = trees.binomial_price(am_call, spot, vol, flat_curve_5pct, val_date, steps=200)

    # Should be close (Tree converges to BS)
    # Expected BS is ~10.45
    assert math.isclose(tree_price, bs_price, rel_tol=1e-2)


def test_american_put_early_exercise(american_put, flat_curve_5pct):
    # Deep ITM Put.
    # Spot = 60, Strike = 100. Intrinsic = 40.
    # Risk free rate = 5%.
    # Holding it to maturity loses time value of money on the 40 profit.
    # Early exercise should be optimal -> Price should be 40.0.
    # Black-Scholes (European) would discount the strike -> Price < 40.

    val_date = date(2025, 1, 1)
    spot = 60.0
    vol = 0.20

    # European Price
    eu_put = instruments.EuropeanOption("TEST", 100.0, american_put.expiry_date, "put")
    bs_price = equity.black_scholes_price(eu_put, spot, vol, flat_curve_5pct, val_date)

    # Tree Price
    tree_price = trees.binomial_price(american_put, spot, vol, flat_curve_5pct, val_date, steps=50)

    print(f"BS: {bs_price}, Tree: {tree_price}")

    # Check 1: American should be worth more (or equal) than European
    assert tree_price > bs_price

    # Check 2: Ideally close to intrinsic 40.0 because waiting is suboptimal
    assert math.isclose(tree_price, 40.0, abs_tol=0.1)


def test_tree_expired_option(american_put, flat_curve_5pct):
    val_date = date(2026, 1, 2)
    price = trees.binomial_price(american_put, 100.0, 0.20, flat_curve_5pct, val_date)
    assert price == 0.0


def test_tree_payoff_at_maturity(american_put, flat_curve_5pct):
    val_date = american_put.expiry_date
    # Spot 90, Strike 100 -> Value 10
    price = trees.binomial_price(american_put, 90.0, 0.20, flat_curve_5pct, val_date)
    assert price == 10.0
