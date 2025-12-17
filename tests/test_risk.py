import math
from datetime import date

import pytest

from pykwant import dates, instruments, rates, risk


def test_shift_curve_flat() -> None:
    ref_date = date(2023, 1, 1)

    def flat_curve(d: date) -> rates.DiscountFactor:
        t = (d - ref_date).days / 365.0
        return rates.DiscountFactor(math.exp(-0.05 * t))

    bump = 0.01
    shifted_curve = risk.shift_curve(flat_curve, bump, ref_date)

    test_date = date(2024, 1, 1)
    t = (test_date - ref_date).days / 365.0

    df_orig = flat_curve(test_date)
    df_new = shifted_curve(test_date)

    expected_factor = math.exp(-bump * t)
    assert df_new == pytest.approx(df_orig * expected_factor)

    implied_rate = -math.log(df_new) / t
    assert implied_rate == pytest.approx(0.06)


def test_risk_metrics_zero_coupon() -> None:
    ref_date = date(2023, 1, 1)
    maturity = date(2024, 1, 1)

    bond = instruments.FixedRateBond(
        face_value=instruments.Money(100.0),
        coupon_rate=0.0,
        start_date=ref_date,
        maturity_date=maturity,
        frequency_months=12,
        day_count=dates.act_365,
    )

    def flat_curve(d: date) -> rates.DiscountFactor:
        t = (d - ref_date).days / 365.0
        return rates.DiscountFactor(math.exp(-0.05 * t))

    metrics = risk.calculate_risk_metrics(bond, flat_curve, ref_date)

    price = metrics["price"]
    duration = metrics["duration"]

    expected_price = 100.0 * math.exp(-0.05)
    assert price == pytest.approx(expected_price)

    assert duration == pytest.approx(1.0, abs=1e-4)

    assert metrics["dv01"] == pytest.approx(duration * price * 0.0001)


def test_risk_metrics_expired_bond() -> None:
    valuation_date = date(2026, 1, 1)
    start = date(2023, 1, 1)
    maturity = date(2025, 1, 1)

    bond = instruments.FixedRateBond(
        face_value=instruments.Money(1000.0),
        coupon_rate=0.05,
        start_date=start,
        maturity_date=maturity,
        frequency_months=12,
    )

    metrics = risk.calculate_risk_metrics(
        bond, lambda d: rates.DiscountFactor(1.0), valuation_date
    )

    assert metrics["price"] == 0.0
    assert metrics["duration"] == 0.0
    assert metrics["convexity"] == 0.0
    assert metrics["dv01"] == 0.0
