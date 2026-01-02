"""
Test suite for pykwant.optimization module.
"""

import math

import pytest

from pykwant import optimization

# --- Fixtures ---


@pytest.fixture
def asset_returns():
    # Asset A: Low risk, Low return (3%)
    # Asset B: High risk, High return (10%)
    return [0.03, 0.10]


@pytest.fixture
def cov_matrix_uncorrelated():
    # Uncorrelated assets
    # Vol A = 5% -> Var = 0.0025
    # Vol B = 20% -> Var = 0.0400
    # Cov = 0
    return [[0.0025, 0.0], [0.0, 0.0400]]


@pytest.fixture
def cov_matrix_correlated():
    # Perfectly correlated assets
    # Vol A = 0.1, Vol B = 0.1
    # Var = 0.01
    # Cov = 0.01 (Rho = 1.0)
    return [[0.01, 0.01], [0.01, 0.01]]


# --- Unit Tests ---


def test_generate_random_weights():
    weights = optimization.generate_random_weights(5)
    assert len(weights) == 5
    assert math.isclose(sum(weights), 1.0)
    assert all(0 <= w <= 1 for w in weights)


def test_portfolio_return(asset_returns):
    # 50/50 weights
    weights = [0.5, 0.5]
    expected = 0.5 * 0.03 + 0.5 * 0.10
    calc = optimization._portfolio_return(weights, asset_returns)
    assert math.isclose(calc, expected)


def test_portfolio_variance_uncorrelated(cov_matrix_uncorrelated):
    # w = [0.5, 0.5]
    # Var = w1^2 * s1^2 + w2^2 * s2^2 (since cov is 0)
    weights = [0.5, 0.5]
    calc = optimization._portfolio_variance(weights, cov_matrix_uncorrelated)

    expected = (0.5**2 * 0.0025) + (0.5**2 * 0.0400)
    assert math.isclose(calc, expected)


def test_portfolio_variance_correlated(cov_matrix_correlated):
    # Perfect correlation, equal volatility 10%
    # Portfolio volatility should be average = 10% -> Variance 0.01
    weights = [0.5, 0.5]
    var = optimization._portfolio_variance(weights, cov_matrix_correlated)
    assert math.isclose(var, 0.01)


# --- Integration Tests (Monte Carlo Optimizer) ---


def test_optimizer_basic(asset_returns, cov_matrix_uncorrelated):
    # Asset A: 3% Return, 5% Vol -> Sharpe ~ 0.6 (Rf=0)
    # Asset B: 10% Return, 20% Vol -> Sharpe ~ 0.5
    # The Max Sharpe portfolio should favor Asset A significantly or mix them for diversification.
    # The Min Vol portfolio should clearly be 100% Asset A (since it has much lower variance).

    max_sharpe, min_vol = optimization.optimize_portfolio_monte_carlo(
        asset_returns, cov_matrix_uncorrelated, num_portfolios=1000, seed=42
    )

    # 1. Min Vol check
    # Min Vol should be heavily weighted towards Asset A (index 0)
    # Since VarA << VarB
    assert min_vol.weights[0] > 0.80
    assert min_vol.volatility < 0.06  # Close to 5%

    # 2. Max Sharpe check
    # Sharpe of A alone is 0.6.
    # A mix can do better due to diversification (correlation 0).
    assert max_sharpe.sharpe_ratio >= 0.6

    # Ensure stats are consistent
    assert max_sharpe.expected_return > 0
    assert max_sharpe.volatility > 0


def test_optimizer_input_validation():
    with pytest.raises(ValueError):
        # Mismatch in dimensions
        optimization.optimize_portfolio_monte_carlo([0.1], [[1, 0], [0, 1]])
