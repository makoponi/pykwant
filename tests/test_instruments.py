import math
from datetime import date

from pykwant.dates import thirty_360
from pykwant.instruments import (
    FixedRateBond,
    Money,
    accrued_interest,
    clean_price,
    generate_cash_flows,
    price_instrument,
)


def test_generate_cash_flows_annual() -> None:
    bond = FixedRateBond(
        face_value=Money(100.0),
        coupon_rate=0.05,
        start_date=date(2025, 1, 1),
        maturity_date=date(2026, 1, 1),
        frequency_months=12,
        day_count=thirty_360,
    )
    flows = generate_cash_flows(bond)

    assert len(flows) == 2

    assert flows[0].type == "coupon"
    assert flows[0].payment_date == date(2026, 1, 1)
    assert math.isclose(flows[0].amount, 5.0)

    assert flows[1].type == "principal"
    assert flows[1].payment_date == date(2026, 1, 1)
    assert math.isclose(flows[1].amount, 100.0)


def test_generate_cash_flows_semiannual() -> None:
    bond = FixedRateBond(
        face_value=Money(1000.0),
        coupon_rate=0.04,  # 4%
        start_date=date(2025, 1, 1),
        maturity_date=date(2026, 1, 1),
        frequency_months=6,
        day_count=thirty_360,
    )
    flows = generate_cash_flows(bond)

    assert len(flows) == 3

    assert flows[0].payment_date == date(2025, 7, 1)
    assert math.isclose(flows[0].amount, 20.0)

    assert flows[1].payment_date == date(2026, 1, 1)
    assert math.isclose(flows[1].amount, 20.0)

    assert flows[2].payment_date == date(2026, 1, 1)
    assert math.isclose(flows[2].amount, 1000.0)


def test_price_instrument_at_par() -> None:
    bond = FixedRateBond(
        face_value=Money(100.0),
        coupon_rate=0.05,
        start_date=date(2025, 1, 1),
        maturity_date=date(2027, 1, 1),
        frequency_months=12,
        day_count=thirty_360,
    )

    def mock_curve(d: date) -> float:
        if d == date(2026, 1, 1):
            return 1.0 / 1.05
        if d == date(2027, 1, 1):
            return 1.0 / (1.05**2)
        return 1.0

    valuation_date = date(2025, 1, 1)

    price = price_instrument(bond, mock_curve, valuation_date)

    assert math.isclose(price, 100.0)


def test_price_instrument_expired() -> None:
    bond = FixedRateBond(
        face_value=Money(100.0),
        coupon_rate=0.05,
        start_date=date(2020, 1, 1),
        maturity_date=date(2021, 1, 1),
        frequency_months=12,
    )

    valuation_date = date(2022, 1, 1)
    price = price_instrument(bond, lambda d: 1.0, valuation_date)
    assert price == 0.0


def test_accrued_interest() -> None:
    bond = FixedRateBond(
        face_value=Money(100.0),
        coupon_rate=0.05,
        start_date=date(2025, 1, 1),
        maturity_date=date(2026, 1, 1),
        frequency_months=12,
        day_count=thirty_360,
    )
    valuation_date = date(2025, 4, 1)

    accrued = accrued_interest(bond, valuation_date)

    assert math.isclose(accrued, 1.25)


def test_clean_price() -> None:
    bond = FixedRateBond(
        face_value=Money(100.0),
        coupon_rate=0.05,
        start_date=date(2025, 1, 1),
        maturity_date=date(2026, 1, 1),
        frequency_months=12,
        day_count=thirty_360,
    )
    valuation_date = date(2025, 7, 1)

    cp = clean_price(bond, lambda d: 1.0, valuation_date)
    assert math.isclose(cp, 102.5)
