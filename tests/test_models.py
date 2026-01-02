"""
Test suite for pykwant.models module.
"""

import math

import pytest

from pykwant import models

# --- Fixtures ---


@pytest.fixture
def flat_curve_wrapper():
    """
    Returns a time-based curve function P(0, t) = exp(-0.05 * t).
    """

    def _curve(t: float) -> float:
        return math.exp(-0.05 * t)

    return _curve


@pytest.fixture
def hw_model():
    # Mean Reversion = 10%, Volatility = 1%
    return models.HullWhiteModel(mean_reversion=0.10, volatility=0.01)


# --- ZCB Pricing Tests ---


def test_zcb_price_at_time_zero(hw_model, flat_curve_wrapper):
    # At t=0, the model Price must match the Market Curve Price exactly (by construction).
    T = 5.0
    # r_0 should be f(0, 0) = 0.05
    r_0 = 0.05

    model_price = models.zcb_price(hw_model, flat_curve_wrapper, r_0, 0.0, T)
    market_price = flat_curve_wrapper(T)

    assert math.isclose(model_price, market_price, rel_tol=1e-9)


def test_zcb_price_maturity(hw_model, flat_curve_wrapper):
    # At t=T, Price should be 1.0
    price = models.zcb_price(hw_model, flat_curve_wrapper, 0.05, 5.0, 5.0)
    assert price == 1.0


# --- Simulation Tests ---


def test_short_rate_mean(hw_model, flat_curve_wrapper):
    """
    Test that the average simulated short rate matches the forward curve.
    E[r(t)] approx f(0, t) + convexity
    """
    T = 1.0
    steps = 10
    paths = models.simulate_short_rate_paths(
        hw_model, flat_curve_wrapper, time_horizon=T, steps=steps, num_paths=5000, seed=42
    )

    # Check mean at terminal step (t=1.0)
    final_rates = [p[-1] for p in paths]
    mean_r = sum(final_rates) / len(final_rates)

    # Analytical Expectation for HW:
    # E[r(t)] = f(0, t) + (sigma^2 / 2a^2) * (1 - exp(-at))^2
    a = hw_model.mean_reversion
    sigma = hw_model.volatility
    convexity = (sigma**2 / (2 * a**2)) * (1.0 - math.exp(-a * T)) ** 2
    expected = 0.05 + convexity

    assert math.isclose(mean_r, expected, rel_tol=0.01)


def test_short_rate_volatility(hw_model, flat_curve_wrapper):
    """
    Test the variance of the short rate.
    Var[r(t)] = sigma^2 / (2a) * (1 - exp(-2at))
    """
    T = 2.0
    paths = models.simulate_short_rate_paths(
        hw_model, flat_curve_wrapper, time_horizon=T, steps=10, num_paths=10000, seed=42
    )

    final_rates = [p[-1] for p in paths]

    # Calculate sample variance
    mean_val = sum(final_rates) / len(final_rates)
    var_sample = sum((r - mean_val) ** 2 for r in final_rates) / (len(final_rates) - 1)

    # Analytical variance
    a = hw_model.mean_reversion
    sigma = hw_model.volatility
    var_analytical = (sigma**2 / (2 * a)) * (1.0 - math.exp(-2 * a * T))

    # Variance convergence is slower, allow 5% tolerance
    assert math.isclose(var_sample, var_analytical, rel_tol=0.05)
