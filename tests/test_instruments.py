"""
Test suite for pykwant.instruments module.
"""

import math
from datetime import date

import pytest

from pykwant import dates, instruments

# --- Fixtures ---


@pytest.fixture
def sample_calendar():
    """Simple calendar with weekends only."""
    return dates.Calendar(holidays=frozenset(), weekends=(6, 7))


@pytest.fixture
def sample_bond(sample_calendar):
    """
    Creates a standard 2-Year Bond.
    Face: 100
    Coupon: 5% (Annual)
    Start: 2025-01-01
    Maturity: 2027-01-01
    """
    return instruments.FixedRateBond(
        face_value=instruments.Money(100.0),
        coupon_rate=0.05,
        start_date=date(2025, 1, 1),
        maturity_date=date(2027, 1, 1),
        frequency_months=12,
        day_count=dates.thirty_360,
        calendar=sample_calendar,
    )


@pytest.fixture
def flat_curve():
    """Flat 5% continuously compounded curve."""
    ref_date = date(2025, 1, 1)

    def _curve(d: date) -> float:
        t = dates.act_365(ref_date, d)
        return math.exp(-0.05 * t)

    return _curve


# --- 1. Data Structure Tests ---


def test_bond_creation(sample_bond):
    assert sample_bond.face_value == 100.0
    assert sample_bond.coupon_rate == 0.05
    assert isinstance(sample_bond.calendar, dates.Calendar)


def test_option_creation():
    opt = instruments.EuropeanOption(
        asset_name="TEST",
        strike=instruments.Money(100.0),
        expiry_date=date(2026, 1, 1),
        call_put="put",
    )
    assert opt.call_put == "put"
    assert opt.asset_name == "TEST"


# --- 2. Cash Flow Generation Tests ---


def test_generate_cash_flows(sample_bond):
    flows = instruments.generate_cash_flows(sample_bond)

    # 2 Years, Annual -> 2 Coupons + 1 Principal = 3 Flows
    assert len(flows) == 3

    # Check First Coupon
    c1 = flows[0]
    assert c1.type == "coupon"
    assert c1.payment_date == date(2026, 1, 1)  # Adjusted if needed, but here simple
    assert math.isclose(c1.amount, 5.0)  # 100 * 0.05 * 1

    # Check Principal
    principal = flows[-1]
    assert principal.type == "principal"
    assert principal.payment_date == date(2027, 1, 1)
    assert principal.amount == 100.0


# --- 3. Accrued Interest Tests ---


def test_accrued_interest_start(sample_bond):
    # At start date, accrued is 0
    val_date = date(2025, 1, 1)
    accrued = instruments.accrued_interest(sample_bond, val_date)
    assert accrued == 0.0


def test_accrued_interest_mid_period(sample_bond):
    # 3 months in (April 1st) using 30/360
    # Jan, Feb, Mar = 3 * 30 = 90 days.
    val_date = date(2025, 4, 1)

    # Accrued = 100 * 0.05 * (90/360) = 5 * 0.25 = 1.25
    accrued = instruments.accrued_interest(sample_bond, val_date)
    assert math.isclose(accrued, 1.25)


def test_accrued_interest_after_coupon(sample_bond):
    # Just after first coupon (Jan 2nd 2026)
    val_date = date(2026, 1, 2)
    # 1 day accrued
    accrued = instruments.accrued_interest(sample_bond, val_date)
    expected = 100 * 0.05 * (1.0 / 360.0)
    assert math.isclose(accrued, expected)


# --- 4. Pricing Tests (Bonds) ---


def test_price_instrument_bond(sample_bond, flat_curve):
    # Valuation at Start
    val_date = date(2025, 1, 1)

    # With 5% Coupon and 5% Curve, Price should be approx Par (~100)
    # Note: Curve is continuous 5%, Coupon is discrete 5%.
    # Continuous discount factors are slightly lower than discrete 1/(1+r).
    # So Price will be slightly different from 100, but close.

    price = instruments.price_instrument(sample_bond, flat_curve, val_date)

    # Calculate expected manually
    # Flow 1 (1Y): 5 * exp(-0.05 * 1)
    # Flow 2 (2Y): 105 * exp(-0.05 * 2)
    t1 = dates.act_365(val_date, date(2026, 1, 1))
    t2 = dates.act_365(val_date, date(2027, 1, 1))

    expected = 5.0 * math.exp(-0.05 * t1) + 105.0 * math.exp(-0.05 * t2)

    assert math.isclose(price, expected, rel_tol=1e-9)


def test_clean_price(sample_bond, flat_curve):
    val_date = date(2025, 4, 1)

    dirty = instruments.price_instrument(sample_bond, flat_curve, val_date)
    accrued = instruments.accrued_interest(sample_bond, val_date)

    clean = instruments.clean_price(sample_bond, flat_curve, val_date)

    assert math.isclose(clean, dirty - accrued)


# --- 5. Pricing Tests (Options - Error Handling) ---


def test_price_instrument_option_raises(flat_curve):
    opt = instruments.EuropeanOption("TEST", instruments.Money(100), date(2026, 1, 1))
    val_date = date(2025, 1, 1)

    # instruments.py only handles bonds. Options should raise NotImplementedError
    # guiding the user to the 'equity' module.
    with pytest.raises(NotImplementedError) as excinfo:
        instruments.price_instrument(opt, flat_curve, val_date)

    assert "equity" in str(excinfo.value)
