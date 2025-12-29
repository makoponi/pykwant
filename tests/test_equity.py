"""
Test suite for pykwant.equity module.
"""

import math
from datetime import date

import pytest

from pykwant import dates, equity, instruments

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
def sample_call():
    """ATM Call Option expiring in 1 Year."""
    return instruments.EuropeanOption(
        asset_name="TEST",
        strike=instruments.Money(100.0),
        expiry_date=date(2026, 1, 1),
        call_put="call",
    )


@pytest.fixture
def sample_put():
    """ATM Put Option expiring in 1 Year."""
    return instruments.EuropeanOption(
        asset_name="TEST",
        strike=instruments.Money(100.0),
        expiry_date=date(2026, 1, 1),
        call_put="put",
    )


# --- Standard Pricing Tests ---


def test_black_scholes_call_value(sample_call, flat_curve_5pct):
    # Benchmark:
    # S=100, K=100, T=1, r=5%, vol=20%
    # d1 = (ln(1) + (0.05 + 0.02) * 1) / 0.20 = 0.07 / 0.20 = 0.35
    # d2 = 0.35 - 0.20 = 0.15
    # N(d1) ~ 0.6368, N(d2) ~ 0.5596
    # Call = 100 * 0.6368 - 100 * e^-0.05 * 0.5596
    # Call ~ 63.68 - 95.12 * 0.5596 ~ 63.68 - 53.23 ~ 10.45

    val_date = date(2025, 1, 1)
    spot = 100.0
    vol = 0.20

    price = equity.black_scholes_price(sample_call, spot, vol, flat_curve_5pct, val_date)

    # Precise benchmark value: 10.45058
    assert math.isclose(price, 10.45058, rel_tol=1e-4)


def test_black_scholes_put_value(sample_put, flat_curve_5pct):
    # Benchmark for Put with same params:
    # Put = Call - S + K * e^-rT
    # Put = 10.4506 - 100 + 100 * 0.9512
    # Put = 10.4506 - 100 + 95.1229 = 5.5735

    val_date = date(2025, 1, 1)
    spot = 100.0
    vol = 0.20

    price = equity.black_scholes_price(sample_put, spot, vol, flat_curve_5pct, val_date)

    # Precise benchmark value: 5.5735
    assert math.isclose(price, 5.5735, rel_tol=1e-4)


# --- Financial Consistency Tests ---


def test_put_call_parity(sample_call, sample_put, flat_curve_5pct):
    """
    Checks the fundamental no-arbitrage relationship:
    Call - Put = Spot - Strike * DF
    """
    val_date = date(2025, 1, 1)
    spot = 100.0
    vol = 0.20

    call_price = equity.black_scholes_price(sample_call, spot, vol, flat_curve_5pct, val_date)
    put_price = equity.black_scholes_price(sample_put, spot, vol, flat_curve_5pct, val_date)

    df = flat_curve_5pct(sample_call.expiry_date)
    strike = sample_call.strike

    lhs = call_price - put_price
    rhs = spot - strike * df

    assert math.isclose(lhs, rhs, rel_tol=1e-9)


# --- Edge Cases Tests ---


def test_expired_option(sample_call, flat_curve_5pct):
    # Valuation date AFTER expiry
    val_date = date(2026, 1, 2)
    spot = 120.0  # ITM

    price = equity.black_scholes_price(sample_call, spot, 0.20, flat_curve_5pct, val_date)
    assert price == 0.0


def test_zero_volatility_call(sample_call, flat_curve_5pct):
    # If Vol=0, Option Value = Max(Fwd - K, 0) * DF
    # essentially intrinsic value of forward.
    val_date = date(2025, 1, 1)
    spot = 100.0
    vol = 0.0

    # Forward = 100 * e^(0.05 * 1) = 105.127
    # Strike = 100
    # Intrinsic Fwd = 5.127
    # PV = 5.127 * e^-0.05 = 5.127 * 0.9512 = 4.87...
    # Actually simpler: Max(S - K*DF, 0)
    # 100 - 100 * 0.9512 = 100 - 95.12 = 4.88

    price = equity.black_scholes_price(sample_call, spot, vol, flat_curve_5pct, val_date)

    df = flat_curve_5pct(sample_call.expiry_date)
    expected = max(spot - 100.0 * df, 0.0)

    assert math.isclose(price, expected, rel_tol=1e-5)


def test_at_expiry_ITM(sample_call, flat_curve_5pct):
    # Valuation exactly at expiry
    val_date = sample_call.expiry_date
    spot = 110.0  # ITM by 10

    price = equity.black_scholes_price(sample_call, spot, 0.20, flat_curve_5pct, val_date)

    # Should be exactly intrinsic value
    assert price == 10.0


def test_at_expiry_OTM(sample_call, flat_curve_5pct):
    # Valuation exactly at expiry
    val_date = sample_call.expiry_date
    spot = 90.0  # OTM

    price = equity.black_scholes_price(sample_call, spot, 0.20, flat_curve_5pct, val_date)

    assert price == 0.0
