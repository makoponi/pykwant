"""
Test suite for pykwant.portfolio module.
"""

import math
from datetime import date

import pytest

from pykwant import dates, instruments, portfolio

# --- Fixtures ---


@pytest.fixture
def sample_calendar():
    return dates.Calendar(holidays=frozenset(), weekends=(6, 7))


@pytest.fixture
def flat_curve_5pct():
    """Flat 5% continuously compounded curve."""
    ref_date = date(2025, 1, 1)

    def _curve(d: date) -> float:
        t = dates.act_365(ref_date, d)
        return math.exp(-0.05 * t)

    return _curve


@pytest.fixture
def bond_short(sample_calendar):
    """A short-term bond (Maturity 2026)."""
    return instruments.FixedRateBond(
        face_value=instruments.Money(100.0),
        coupon_rate=0.05,
        start_date=date(2025, 1, 1),
        maturity_date=date(2026, 1, 1),  # 1 Year
        frequency_months=12,
        day_count=dates.thirty_360,
        calendar=sample_calendar,
    )


@pytest.fixture
def bond_long(sample_calendar):
    """A long-term bond (Maturity 2030)."""
    return instruments.FixedRateBond(
        face_value=instruments.Money(100.0),
        coupon_rate=0.05,
        start_date=date(2025, 1, 1),
        maturity_date=date(2030, 1, 1),  # 5 Years
        frequency_months=12,
        day_count=dates.thirty_360,
        calendar=sample_calendar,
    )


@pytest.fixture
def sample_portfolio(bond_short, bond_long):
    """
    Portfolio:
    - Long 10 units of Short Bond (1Y)
    - Long 5 units of Long Bond (5Y)
    - Short 2 units of Short Bond (Netting test)
    """
    return [
        portfolio.Position(bond_short, 10.0),
        portfolio.Position(bond_long, 5.0),
        portfolio.Position(bond_short, -2.0),  # Net position for bond_short is 8.0
    ]


# --- Tests ---


def test_portfolio_npv(sample_portfolio, flat_curve_5pct, bond_short, bond_long):
    val_date = date(2025, 1, 1)

    # Calculate Prices Individually
    p_short = instruments.price_instrument(bond_short, flat_curve_5pct, val_date)
    p_long = instruments.price_instrument(bond_long, flat_curve_5pct, val_date)

    # Expected NPV = (10 - 2) * p_short + 5 * p_long
    expected_npv = 8.0 * p_short + 5.0 * p_long

    calculated_npv = portfolio.portfolio_npv(sample_portfolio, flat_curve_5pct, val_date)

    assert math.isclose(calculated_npv, expected_npv)


def test_portfolio_risk_aggregation(sample_portfolio, flat_curve_5pct):
    val_date = date(2025, 1, 1)

    risk_report = portfolio.portfolio_risk(sample_portfolio, flat_curve_5pct, val_date)

    # 1. Market Value Check
    npv = portfolio.portfolio_npv(sample_portfolio, flat_curve_5pct, val_date)
    assert math.isclose(risk_report["market_value"], npv)

    # 2. Total DV01 Check
    # DV01 is additive. We assume individual calculations are correct
    # (tested in test_risk).
    # Since prices fall as rates rise, DV01 should be negative for a net long portfolio.
    assert risk_report["total_dv01"] < 0.0

    # 3. Portfolio Duration Check
    # It must be positive and between the duration of the 1Y bond (~1.0)
    # and 5Y bond (~4.5).
    # Since we have more weight in the 1Y bond (800 vs 500 roughly),
    # It should be closer to 1Y.
    port_dur = risk_report["portfolio_duration"]
    assert 1.0 < port_dur < 4.5


def test_exposure_by_maturity_year(sample_portfolio, flat_curve_5pct, bond_short, bond_long):
    val_date = date(2025, 1, 1)

    exposure = portfolio.exposure_by_maturity_year(sample_portfolio, flat_curve_5pct, val_date)

    p_short = instruments.price_instrument(bond_short, flat_curve_5pct, val_date)
    p_long = instruments.price_instrument(bond_long, flat_curve_5pct, val_date)

    # Expected:
    # 2026: 8 units * p_short
    # 2030: 5 units * p_long

    assert math.isclose(exposure[2026], 8.0 * p_short)
    assert math.isclose(exposure[2030], 5.0 * p_long)

    # Verify no other years are present
    assert len(exposure) == 2


def test_empty_portfolio(flat_curve_5pct):
    val_date = date(2025, 1, 1)
    empty_port: list[portfolio.Position] = []

    npv = portfolio.portfolio_npv(empty_port, flat_curve_5pct, val_date)
    assert npv == 0.0

    risk_report = portfolio.portfolio_risk(empty_port, flat_curve_5pct, val_date)
    assert risk_report["market_value"] == 0.0
    assert risk_report["portfolio_duration"] == 0.0

    exposure = portfolio.exposure_by_maturity_year(empty_port, flat_curve_5pct, val_date)
    assert exposure == {}
