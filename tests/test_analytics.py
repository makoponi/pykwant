"""
Test suite for pykwant.analytics module.
"""

import math

import pytest

from pykwant import analytics

# --- Fixtures ---


@pytest.fixture
def prices_uptrend():
    return [100.0, 101.0, 102.01, 103.0301]  # 1% steady growth


@pytest.fixture
def prices_volatile():
    return [100.0, 90.0, 95.0, 80.0, 110.0]


# --- Returns Tests ---


def test_simple_returns(prices_uptrend):
    rets = analytics.simple_returns(prices_uptrend)
    assert len(rets) == 3
    # 101/100 - 1 = 0.01
    assert math.isclose(rets[0], 0.01)
    assert math.isclose(rets[1], 0.01)


def test_log_returns(prices_uptrend):
    rets = analytics.log_returns(prices_uptrend)
    # ln(1.01) ~ 0.00995
    expected = math.log(1.01)
    assert math.isclose(rets[0], expected)


def test_returns_empty():
    assert analytics.simple_returns([100.0]) == []
    assert analytics.simple_returns([]) == []


# --- Drawdown Tests ---


def test_max_drawdown_none(prices_uptrend):
    # No drawdown in uptrend
    mdd = analytics.max_drawdown(prices_uptrend)
    assert mdd == 0.0


def test_max_drawdown_volatile(prices_volatile):
    # Sequence: 100 -> 90 (Peak 100, DD 10%)
    # 90 -> 95 (Peak 100, DD 5%)
    # 95 -> 80 (Peak 100, DD 20%) -> Max so far
    # 80 -> 110 (New Peak 110, DD 0%)
    mdd = analytics.max_drawdown(prices_volatile)
    assert math.isclose(mdd, 0.20)


def test_max_drawdown_crash():
    prices = [100.0, 50.0, 25.0]
    # 100 -> 25 is 75% drop
    assert math.isclose(analytics.max_drawdown(prices), 0.75)


# --- Volatility & Sharpe Tests ---


def test_annualized_volatility():
    # Returns: +1%, -1%. Sample StdDev.
    # Mean = 0.
    # Var = ((0.01-0)^2 + (-0.01-0)^2) / (2-1) = 0.0001 + 0.0001 = 0.0002
    # Std = sqrt(0.0002) ~ 0.01414
    # Ann = 0.01414 * sqrt(252) ~ 0.01414 * 15.87 ~ 0.2245
    rets = [0.01, -0.01]
    vol = analytics.annualized_volatility(rets, 252)

    std_p = math.sqrt(0.0002)
    expected = std_p * math.sqrt(252)
    assert math.isclose(vol, expected)


def test_sharpe_ratio():
    # Stable 1% return daily, Rf=0.
    # StdDev = 0. Sharpe -> Inf/Undefined in reality, but if small noise?
    # Let's use [0.01, 0.02].
    # Mean = 0.015.
    # Std ~ 0.00707
    # Sharpe = (0.015 / 0.00707) * sqrt(252) ~ 2.12 * 15.87 ~ 33.6
    rets = [0.01, 0.02]
    sharpe = analytics.sharpe_ratio(rets)

    avg = 0.015
    std = math.sqrt(
        ((0.01 - avg) ** 2 + (0.02 - avg) ** 2) / 1
    )  # sqrt(0.000025 + 0.000025) -> 0.007071
    expected = (avg / std) * math.sqrt(252)

    assert math.isclose(sharpe, expected)


def test_sharpe_with_rf():
    # Daily returns = 0.01. Annual Rf = 0.02 -> Daily Rf ~ 0.02/252 ~ 0.00008
    # Excess ~ 0.0099.
    rets = [0.01, 0.02]
    rf = 0.02
    sharpe = analytics.sharpe_ratio(rets, risk_free_rate=rf)

    # Recalculate manually
    avg = 0.015
    rf_p = 0.02 / 252
    excess = avg - rf_p
    std = math.sqrt(((0.01 - avg) ** 2 + (0.02 - avg) ** 2) / 1)
    expected = (excess / std) * math.sqrt(252)

    assert math.isclose(sharpe, expected)


def test_sortino_ratio():
    # Returns: [0.05, -0.02]. Target=0.
    # Mean = 0.015.
    # Downside: only -0.02 is < 0.
    # Downside Sq sum = (-0.02)^2 = 0.0004.
    # Downside Dev (N-1=1) = sqrt(0.0004) = 0.02.
    # Sortino = (0.015 / 0.02) * sqrt(252) = 0.75 * 15.874 = 11.905
    rets = [0.05, -0.02]
    sortino = analytics.sortino_ratio(rets)

    expected = (0.015 / 0.02) * math.sqrt(252)
    assert math.isclose(sortino, expected)
