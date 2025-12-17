import math
from datetime import date, timedelta

import pytest

from pykwant import dates, rates


def test_create_curve_from_discount_factor() -> None:
    ref_date = date(2023, 1, 1)
    dates_list = [
        ref_date + timedelta(days=365),  # Year 1
        ref_date + timedelta(days=730),  # Year 2
    ]

    r = 0.05
    dfs = [
        math.exp(-r * 1.0),
        math.exp(-r * 2.0),
    ]

    curve = rates.create_curve_from_discount_factor(
        ref_date, dates_list, dfs, day_count=dates.act_365
    )

    assert curve(dates_list[0]) == pytest.approx(dfs[0])
    assert curve(dates_list[1]) == pytest.approx(dfs[1])
    assert curve(ref_date) == pytest.approx(1.0)

    mid_date = ref_date + timedelta(days=int(365 * 1.5))
    df_mid = curve(mid_date)
    expected_mid = math.exp(-r * 1.5)

    assert df_mid == pytest.approx(expected_mid, rel=1e-3)


def test_zero_rates() -> None:
    ref_date = date(2023, 1, 1)
    target_date = date(2024, 1, 1)

    def mock_curve(d: date) -> rates.DiscountFactor:
        return rates.DiscountFactor(0.95)

    t = dates.act_365(ref_date, target_date)
    expected_rate = -math.log(0.95) / t

    z_rate = rates.zero_rates(mock_curve, ref_date, target_date)
    assert z_rate == pytest.approx(expected_rate)

    assert rates.zero_rates(mock_curve, ref_date, ref_date) == 0.0


def test_forward_rate() -> None:
    ref_date = date(2023, 1, 1)
    start_date = date(2023, 6, 1)
    end_date = date(2024, 6, 1)

    def mock_curve(d: date) -> rates.DiscountFactor:
        t = (d - ref_date).days / 365.0
        return rates.DiscountFactor(math.exp(-0.05 * t))

    df_start = mock_curve(start_date)
    df_end = mock_curve(end_date)
    tau = dates.act_365(start_date, end_date)

    expected_fwd = (df_start / df_end - 1.0) / tau
    calc_fwd = rates.forward_rate(mock_curve, start_date, end_date)

    assert calc_fwd == pytest.approx(expected_fwd)

    assert rates.forward_rate(mock_curve, start_date, start_date) == 0.0


def test_present_value() -> None:
    payment_date = date(2024, 1, 1)

    def mock_curve(d: date) -> rates.DiscountFactor:
        return rates.DiscountFactor(0.8)

    amount = 100.0
    pv = rates.present_value(amount, payment_date, mock_curve)
    assert pv == pytest.approx(80.0)


def test_compound_factor() -> None:
    r = rates.Rate(0.05)
    t = 2.0

    assert rates.compound_factor(r, t, frequency=0) == pytest.approx(math.exp(r * t))

    assert rates.compound_factor(r, t, frequency=1) == pytest.approx((1 + r) ** t)

    assert rates.compound_factor(r, t, frequency=2) == pytest.approx(
        (1 + r / 2) ** (2 * t)
    )
