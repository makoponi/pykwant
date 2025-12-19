"""
Test suite for pykwant.rates module.
"""

import math
from datetime import date

import pytest

from pykwant import dates, rates

# --- 1. Basic Math Tests ---


def test_compound_factor_continuous():
    # e^(rt)
    rate = 0.05
    t = 2.0
    expected = math.exp(rate * t)
    assert math.isclose(rates.compound_factor(rate, t, frequency=0), expected)


def test_compound_factor_discrete():
    # (1 + r/f)^(ft)
    rate = 0.05
    t = 2.0
    freq = 1  # Annual
    expected = (1 + 0.05) ** 2
    assert math.isclose(rates.compound_factor(rate, t, frequency=freq), expected)

    freq = 2  # Semi-annual
    expected = (1 + 0.05 / 2) ** (2 * 2)
    assert math.isclose(rates.compound_factor(rate, t, frequency=freq), expected)


# --- 2. Curve Construction Tests ---


@pytest.fixture
def sample_curve():
    """
    Creates a simple curve from discount factors.
    Ref Date: 2025-01-01
    1Y (2026-01-01): DF = 0.95 (approx 5%)
    2Y (2027-01-01): DF = 0.90 (approx 5.2%)
    """
    ref_date = date(2025, 1, 1)
    dates_list = [date(2026, 1, 1), date(2027, 1, 1)]
    dfs = [0.95, 0.90]

    return rates.create_curve_from_discount_factor(
        reference_date=ref_date,
        dates_list=dates_list,
        dfs_list=dfs,
        day_count=dates.act_365,
    )


def test_curve_interpolation(sample_curve):
    ref_date = date(2025, 1, 1)

    # 1. Check Pillars (Exact match)
    assert math.isclose(sample_curve(date(2026, 1, 1)), 0.95)
    assert math.isclose(sample_curve(date(2027, 1, 1)), 0.90)

    # 2. Check Reference Date (Should be 1.0)
    assert sample_curve(ref_date) == 1.0

    # 3. Check Interpolation (Log-Linear)
    # Midpoint between 1Y and 2Y
    d_mid = date(2026, 7, 2)  # Approx 1.5Y

    # Expected: exp( linear_interp( log(0.95), log(0.90) ) )
    log_df1 = math.log(0.95)
    log_df2 = math.log(0.90)
    expected_log = (log_df1 + log_df2) / 2
    expected_df = math.exp(expected_log)

    # Allow small tolerance for date arithmetic differences
    assert math.isclose(sample_curve(d_mid), expected_df, rel_tol=1e-3)


# --- 3. Rate Helpers Tests ---


def test_zero_rates(sample_curve):
    ref_date = date(2025, 1, 1)
    target_date = date(2026, 1, 1)  # 1 Year

    # r = -ln(DF) / t
    # t is approx 1.0 (365/365)
    df = 0.95
    t = dates.act_365(ref_date, target_date)
    expected_r = -math.log(df) / t

    calc_r = rates.zero_rates(sample_curve, ref_date, target_date)
    assert math.isclose(calc_r, expected_r)


def test_zero_rate_at_ref_date(sample_curve):
    # Should handle division by zero gracefully
    ref_date = date(2025, 1, 1)
    assert rates.zero_rates(sample_curve, ref_date, ref_date) == 0.0


def test_forward_rate(sample_curve):
    # Forward between 1Y and 2Y
    d1 = date(2026, 1, 1)  # DF = 0.95
    d2 = date(2027, 1, 1)  # DF = 0.90

    # F = (DF1 / DF2 - 1) / tau
    tau = dates.act_365(d1, d2)  # approx 1.0
    expected_fwd = (0.95 / 0.90 - 1) / tau

    calc_fwd = rates.forward_rate(sample_curve, d1, d2)
    assert math.isclose(calc_fwd, expected_fwd)


def test_present_value(sample_curve):
    # PV = Amount * DF
    pay_date = date(2026, 1, 1)  # DF = 0.95
    amount = 100.0

    pv = rates.present_value(amount, pay_date, sample_curve)
    assert math.isclose(pv, 95.0)
