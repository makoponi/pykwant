import math
from datetime import date

import pytest

from pykwant import dates, instruments, portfolio, rates


def get_flat_curve(rate_val: float, ref_date: date) -> rates.YieldCurveFn:
    def _curve(d: date) -> rates.DiscountFactor:
        t = (d - ref_date).days / 365.0
        return rates.DiscountFactor(math.exp(-rate_val * t))

    return _curve


def create_bond(maturity_year: int, coupon: float = 0.05) -> instruments.FixedRateBond:
    return instruments.FixedRateBond(
        face_value=instruments.Money(100.0),
        coupon_rate=coupon,
        start_date=date(2023, 1, 1),
        maturity_date=date(maturity_year, 1, 1),
        frequency_months=12,
        day_count=dates.act_365,
    )


def test_evaluate_position() -> None:
    ref_date = date(2023, 1, 1)
    curve = get_flat_curve(0.05, ref_date)
    bond = create_bond(2024)

    quantity = 10.0
    pos = portfolio.Position(instrument=bond, quantity=quantity)

    unit_price = instruments.price_instrument(bond, curve, ref_date)
    value = portfolio.evaluate_position(pos, curve, ref_date)

    assert value == pytest.approx(unit_price * quantity)
    assert pos.direction == 1


def test_portfolio_npv() -> None:
    ref_date = date(2023, 1, 1)
    curve = get_flat_curve(0.05, ref_date)

    bond1 = create_bond(2024)
    bond2 = create_bond(2025)

    port = [
        portfolio.Position(bond1, 100.0),
        portfolio.Position(bond2, -50.0),  # Short position
    ]

    npv = portfolio.portfolio_npv(port, curve, ref_date)

    val1 = portfolio.evaluate_position(port[0], curve, ref_date)
    val2 = portfolio.evaluate_position(port[1], curve, ref_date)

    assert npv == pytest.approx(val1 + val2)
    assert port[1].direction == -1


def test_portfolio_risk_aggregation() -> None:
    ref_date = date(2023, 1, 1)
    curve = get_flat_curve(0.05, ref_date)

    zcb = instruments.FixedRateBond(
        face_value=instruments.Money(100.0),
        coupon_rate=0.0,
        start_date=ref_date,
        maturity_date=date(2024, 1, 1),
        frequency_months=12,
        day_count=dates.act_365,
    )

    pos = portfolio.Position(zcb, 100.0)
    port = [pos]

    risk_res = portfolio.portfolio_risk(port, curve, ref_date)

    mv = risk_res["total_market_value"]
    dv01 = risk_res["total_dv01"]
    dur = risk_res["portfolio_duration"]

    expected_price = 100.0 * math.exp(-0.05)
    expected_mv = expected_price * 100.0

    assert mv == pytest.approx(expected_mv)
    assert dur == pytest.approx(1.0, abs=1e-3)
    # DV01 = Duration * Price * 0.0001 * Quantity
    assert dv01 == pytest.approx(dur * mv * 0.0001)
    assert risk_res["positions_count"] == 1


def test_exposure_by_maturity_year() -> None:
    ref_date = date(2023, 1, 1)
    curve = get_flat_curve(0.05, ref_date)

    bond24 = create_bond(2024)
    bond25_a = create_bond(2025)
    bond25_b = create_bond(2025)

    port = [
        portfolio.Position(bond24, 10.0),
        portfolio.Position(bond25_a, 10.0),
        portfolio.Position(bond25_b, 20.0),
    ]

    exposure = portfolio.exposure_by_maturity_year(port, curve, ref_date)

    val24 = portfolio.evaluate_position(port[0], curve, ref_date)
    val25_a = portfolio.evaluate_position(port[1], curve, ref_date)
    val25_b = portfolio.evaluate_position(port[2], curve, ref_date)

    assert 2024 in exposure
    assert 2025 in exposure
    assert exposure[2024] == pytest.approx(val24)
    assert exposure[2025] == pytest.approx(val25_a + val25_b)
