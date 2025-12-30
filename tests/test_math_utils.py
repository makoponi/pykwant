"""
Test suite for pykwant.math_utils module.
"""

import math

import pytest

from pykwant import math_utils

# --- 1. Gaussian Functions (Normal Distribution) ---


def test_norm_pdf_at_zero():
    # Standard Normal PDF at x=0 is 1/sqrt(2*pi)
    expected = 1.0 / math.sqrt(2.0 * math.pi)
    assert math.isclose(math_utils.norm_pdf(0.0), expected, rel_tol=1e-9)


def test_norm_pdf_symmetry():
    # PDF(x) == PDF(-x)
    x = 1.5
    val_pos = math_utils.norm_pdf(x)
    val_neg = math_utils.norm_pdf(-x)
    assert math.isclose(val_pos, val_neg, rel_tol=1e-9)


def test_norm_cdf_standard_sigmas():
    # 1 Sigma (approx 84.13%)
    assert math.isclose(math_utils.norm_cdf(1.0), 0.8413447, rel_tol=1e-5)
    # 2 Sigma (approx 97.72%)
    assert math.isclose(math_utils.norm_cdf(2.0), 0.9772498, rel_tol=1e-5)


def test_norm_ppf_basics():
    # PPF(0.5) = 0.0 (Mean)
    assert math.isclose(math_utils.norm_ppf(0.5), 0.0, abs_tol=1e-9)

    # PPF(0.975) ~ 1.96 (95% Confidence two-sided)
    assert math.isclose(math_utils.norm_ppf(0.975), 1.959964, rel_tol=1e-5)

    # PPF(0.025) ~ -1.96
    assert math.isclose(math_utils.norm_ppf(0.025), -1.959964, rel_tol=1e-5)


def test_norm_ppf_inversion():
    # CDF(PPF(p)) == p
    for p in [0.01, 0.25, 0.75, 0.99]:
        z = math_utils.norm_ppf(p)
        cdf_val = math_utils.norm_cdf(z)
        assert math.isclose(cdf_val, p, rel_tol=1e-5)


def test_norm_ppf_errors():
    with pytest.raises(ValueError):
        math_utils.norm_ppf(0.0)  # Invalid
    with pytest.raises(ValueError):
        math_utils.norm_ppf(1.0)  # Invalid


# --- 2. Descriptive Statistics Tests ---


def test_mean():
    data = [1.0, 2.0, 3.0, 4.0]
    assert math_utils.mean(data) == 2.5

    with pytest.raises(ValueError):
        math_utils.mean([])


def test_variance():
    # Population: [2, 2] -> Var=0
    assert math_utils.variance([2.0, 2.0], is_sample=False) == 0.0

    # Sample: [1, 2, 3]. Mean=2.
    # DiffSq: 1+0+1 = 2. Divisor (N-1) = 2. Var=1.
    assert math_utils.variance([1.0, 2.0, 3.0], is_sample=True) == 1.0

    # Population: Divisor N=3. Var=2/3 ~ 0.666...
    assert math.isclose(math_utils.variance([1.0, 2.0, 3.0], is_sample=False), 2.0 / 3.0)


def test_std_dev():
    data = [1.0, 2.0, 3.0]
    assert math_utils.std_dev(data, is_sample=True) == 1.0


# --- 3. Relationships Tests ---


def test_covariance():
    # Perfectly correlated
    x = [1.0, 2.0, 3.0]
    y = [2.0, 4.0, 6.0]  # y = 2x

    # Cov(X, Y) = E[(X-Mx)(Y-My)]
    # X-Mx: -1, 0, 1
    # Y-My: -2, 0, 2
    # Prod: 2, 0, 2 -> Sum=4
    # Sample Divisor (N-1) = 2. Cov = 2.0.
    assert math_utils.covariance(x, y, is_sample=True) == 2.0


def test_covariance_errors():
    with pytest.raises(ValueError):
        math_utils.covariance([1, 2], [1, 2, 3])


def test_correlation():
    x = [1.0, 2.0, 3.0]
    y = [2.0, 4.0, 6.0]
    # Perfect positive correlation
    assert math.isclose(math_utils.correlation(x, y), 1.0)

    z = [-2.0, -4.0, -6.0]
    # Perfect negative correlation
    assert math.isclose(math_utils.correlation(x, z), -1.0)

    # Zero correlation (orthogonal)
    # x: 1, 2, 3 (Mean 2). centered: -1, 0, 1
    # w: 1, -2, 1 (Mean 0). centered: 1, -2, 1
    # Prod: -1, 0, 1 -> Sum 0.
    w = [1.0, -2.0, 1.0]
    assert math.isclose(math_utils.correlation(x, w), 0.0)


# --- 4. Percentile Tests ---


def test_percentile_basics():
    data = [15.0, 20.0, 35.0, 40.0, 50.0]
    # Sorted: 15, 20, 35, 40, 50
    # N=5.

    # 0th percentile -> min
    assert math_utils.percentile(data, 0.0) == 15.0

    # 100th percentile -> max
    assert math_utils.percentile(data, 1.0) == 50.0

    # Median (50th) -> 0.5 * (5-1) = 2.0 -> index 2 -> 35.0
    assert math_utils.percentile(data, 0.5) == 35.0


def test_percentile_interpolation():
    data = [10.0, 20.0]
    # N=2. p=0.5 -> k=0.5.
    # Interp between index 0 and 1.
    assert math_utils.percentile(data, 0.5) == 15.0


def test_percentile_unsorted():
    data = [20.0, 10.0]
    # Should sort internally to [10, 20]
    assert math_utils.percentile(data, 0.0) == 10.0
    assert math_utils.percentile(data, 1.0) == 20.0
