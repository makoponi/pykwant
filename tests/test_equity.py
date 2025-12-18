import math
from datetime import date

import pytest

from pykwant import dates, equity, instruments, rates


def test_black_scholes_call_price() -> None:
    """Test pricing of a Call option against known Black-Scholes value."""
    val_date = date(2023, 1, 1)
    expiry_date = date(2024, 1, 1)

    spot = 100.0
    strike = 100.0
    vol = 0.2
    r = 0.05

    option = instruments.EuropeanOption(
        asset_name="TEST_CALL",
        strike=instruments.Money(strike),
        expiry_date=expiry_date,
        type="call",
    )

    def flat_curve(d: date) -> rates.DiscountFactor:
        t = dates.act_365(val_date, d)
        return rates.DiscountFactor(math.exp(-r * t))

    price = equity.black_scholes_price(option, spot, vol, flat_curve, val_date)

    assert price == pytest.approx(10.4506, abs=1e-4)


def test_black_scholes_put_price() -> None:
    """Test pricing of a Put option against known Black-Scholes value."""
    val_date = date(2023, 1, 1)
    expiry_date = date(2024, 1, 1)

    spot = 100.0
    strike = 100.0
    vol = 0.2
    r = 0.05

    option = instruments.EuropeanOption(
        asset_name="TEST_PUT",
        strike=instruments.Money(strike),
        expiry_date=expiry_date,
        type="put",
    )

    def flat_curve(d: date) -> rates.DiscountFactor:
        t = dates.act_365(val_date, d)
        return rates.DiscountFactor(math.exp(-r * t))

    price = equity.black_scholes_price(option, spot, vol, flat_curve, val_date)

    assert price == pytest.approx(5.5735, abs=1e-4)


def test_black_scholes_expiry_logic() -> None:
    """Test che verifica il comportamento a scadenza e post-scadenza."""
    val_date = date(2024, 1, 1)
    expiry_date = date(2024, 1, 1)
    option = instruments.EuropeanOption(
        "TEST", instruments.Money(100.0), expiry_date, "call"
    )

    assert (
        equity.black_scholes_price(
            option, 110.0, 0.2, lambda d: rates.DiscountFactor(1.0), val_date
        )
        == 10.0
    )

    post_expiry = date(2024, 1, 2)
    assert (
        equity.black_scholes_price(
            option, 110.0, 0.2, lambda d: rates.DiscountFactor(1.0), post_expiry
        )
        == 0.0
    )
