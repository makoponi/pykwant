"""
Test suite for pykwant.risk module.
"""

import math
from datetime import date

import pytest

from pykwant import dates, instruments, risk

# --- Fixtures ---


@pytest.fixture
def sample_calendar():
    return dates.Calendar(holidays=frozenset(), weekends=(6, 7))


@pytest.fixture
def par_bond(sample_calendar):
    """
    5-Year Par Bond.
    Coupon: 5%
    Yield: 5% (implied by flat curve)
    Price should be ~100.
    """
    return instruments.FixedRateBond(
        face_value=instruments.Money(100.0),
        coupon_rate=0.05,
        start_date=date(2025, 1, 1),
        maturity_date=date(2030, 1, 1),
        frequency_months=12,
        day_count=dates.thirty_360,
        calendar=sample_calendar,
    )


@pytest.fixture
def zero_coupon_bond(sample_calendar):
    """
    2-Year Zero Coupon Bond.
    Duration should be exactly 2.0.
    """
    return instruments.FixedRateBond(
        face_value=instruments.Money(100.0),
        coupon_rate=0.0,
        start_date=date(2025, 1, 1),
        maturity_date=date(2027, 1, 1),
        frequency_months=12,  # Irrelevant for zero, but required
        day_count=dates.act_365,
        calendar=sample_calendar,
    )


@pytest.fixture
def flat_curve_5pct():
    """Flat 5% continuously compounded curve."""
    ref_date = date(2025, 1, 1)

    def _curve(d: date) -> float:
        t = dates.act_365(ref_date, d)
        return math.exp(-0.05 * t)

    return _curve


# --- Tests ---


def test_pv01_direction(par_bond, flat_curve_5pct):
    val_date = date(2025, 1, 1)

    # Calculate PV01 (Change for +1bp)
    val = risk.pv01(par_bond, flat_curve_5pct, val_date)

    # Prices fall when rates rise -> PV01 should be negative
    assert val < 0.0

    # Magnitude check: ~ 4.4 years duration * 100 price * 1bp = 0.044
    assert math.isclose(abs(val), 0.044, abs_tol=0.01)


def test_effective_duration_zero_coupon(zero_coupon_bond, flat_curve_5pct):
    val_date = date(2025, 1, 1)

    # For a Zero Coupon Bond, Macaulay Duration = Maturity.
    # Modified Duration = Maturity / (1+r) (discrete) or Maturity (continuous).
    # Since our curve is continuous and our shift logic applies to the continuous rate,
    # the derivative dP/dr / P should be exactly -Maturity.

    dur = risk.effective_duration(zero_coupon_bond, flat_curve_5pct, val_date)

    # Maturity is 2 years
    assert math.isclose(dur, 2.0, rel_tol=1e-3)


def test_effective_duration_par_bond(par_bond, flat_curve_5pct):
    val_date = date(2025, 1, 1)

    # 5Y Bond, 5% Coupon, 5% Yield.
    # Duration should be roughly between 4.3 and 4.5
    dur = risk.effective_duration(par_bond, flat_curve_5pct, val_date)

    assert 4.3 < dur < 4.6


def test_effective_convexity_positive(par_bond, flat_curve_5pct):
    val_date = date(2025, 1, 1)

    # Plain vanilla bonds always have positive convexity
    conv = risk.effective_convexity(par_bond, flat_curve_5pct, val_date)

    assert conv > 0.0


def test_calculate_risk_metrics_integration(par_bond, flat_curve_5pct):
    val_date = date(2025, 1, 1)

    metrics = risk.calculate_risk_metrics(par_bond, flat_curve_5pct, val_date)

    assert "price" in metrics
    assert "duration" in metrics
    assert "convexity" in metrics
    assert "dv01" in metrics

    # Check consistency: DV01 approx -Duration * Price * 0.0001
    predicted_dv01 = -metrics["duration"] * metrics["price"] * 0.0001
    assert math.isclose(metrics["dv01"], predicted_dv01, rel_tol=1e-2)


def test_zero_price_instrument(flat_curve_5pct, sample_calendar):
    # Instrument that hasn't started yet or has 0 flows left
    # (Here we hack it by valuing past maturity)
    bond = instruments.FixedRateBond(
        instruments.Money(100.0),
        0.05,
        date(2020, 1, 1),
        date(2021, 1, 1),
        12,
        dates.act_365,
        sample_calendar,
    )
    val_date = date(2025, 1, 1)  # Way past maturity

    # Price is 0
    # Duration should be 0 (and not crash with division by zero)
    dur = risk.effective_duration(bond, flat_curve_5pct, val_date)
    assert dur == 0.0

    conv = risk.effective_convexity(bond, flat_curve_5pct, val_date)
    assert conv == 0.0
