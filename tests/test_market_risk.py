"""
Test suite for pykwant.market_risk module.
"""

import math

import pytest

from pykwant import market_risk, math_utils

# --- Fixtures ---


@pytest.fixture
def sample_returns():
    # 100 returns.
    # Let's create a range -0.05 to +0.049 with step 0.001
    # Sorted indices 0..4 are -0.05, -0.049, -0.048, -0.047, -0.046
    return [0.001 * i - 0.05 for i in range(100)]


# --- Parametric VaR Tests ---


def test_parametric_var_95():
    # Value 1M, Vol 20% Annual, 1 Day Horizon
    # Sigma_daily = 0.20 / sqrt(252) ~ 0.01259
    # z_95 ~ 1.645
    val = 1_000_000
    vol = 0.20
    var = market_risk.parametric_var(val, vol, confidence_level=0.95)

    sigma_day = 0.20 / math.sqrt(252)
    expected = val * sigma_day * 1.644853

    assert math.isclose(var, expected, rel_tol=1e-4)


def test_parametric_var_99():
    # z_99 ~ 2.326
    val = 1_000_000
    vol = 0.20
    var = market_risk.parametric_var(val, vol, confidence_level=0.99)

    sigma_day = 0.20 / math.sqrt(252)
    expected = val * sigma_day * 2.326348

    assert math.isclose(var, expected, rel_tol=1e-4)


# --- Historical VaR Tests ---


def test_historical_var_basic(sample_returns):
    # N=100. 95% Confidence -> 5th percentile.
    # index k = 99 * 0.05 = 4.95.
    # Interp between index 4 (-0.046) and 5 (-0.045).
    # Expected approx -0.04505

    val = 1_000_000
    var = market_risk.historical_var(val, sample_returns, confidence_level=0.95)

    # Percentile is approx -0.04505
    # VaR should be approx 45,050
    expected_loss_pct = abs(-0.046 + 0.95 * (-0.045 - -0.046))

    assert math.isclose(var, val * expected_loss_pct, rel_tol=1e-4)


def test_historical_var_empty():
    with pytest.raises(ValueError):
        market_risk.historical_var(100, [])


# --- Expected Shortfall Tests ---


def test_parametric_es_95():
    # ES = V * sigma * (pdf(z) / (1-c))
    val = 1_000_000
    vol = 0.20

    es = market_risk.parametric_expected_shortfall(val, vol, confidence_level=0.95)

    sigma_day = 0.20 / math.sqrt(252)
    z = -1.644853
    pdf_z = math_utils.norm_pdf(z)
    alpha = 0.05

    expected = val * sigma_day * (pdf_z / alpha)

    assert math.isclose(es, expected, rel_tol=1e-4)

    # Sanity Check: ES should always be > VaR
    var = market_risk.parametric_var(val, vol, confidence_level=0.95)
    assert es > var


def test_historical_es_basic(sample_returns):
    # Cutoff approx -0.04505.
    # Tail returns <= cutoff: [-0.05, -0.049, -0.048, -0.047, -0.046].
    # Avg: -0.048.

    val = 100_000
    es = market_risk.historical_expected_shortfall(val, sample_returns, confidence_level=0.95)

    # Expected approx 100k * 0.048 = 4800
    assert math.isclose(es, val * 0.048, rel_tol=1e-4)

    var = market_risk.historical_var(val, sample_returns, confidence_level=0.95)
    assert es >= var
