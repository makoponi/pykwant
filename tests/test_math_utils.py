"""
Test suite for pykwant.math_utils module.
"""

import math

from pykwant import math_utils

# --- Probability Density Function (PDF) Tests ---


def test_norm_pdf_at_zero():
    # Standard Normal PDF at x=0 is 1/sqrt(2*pi)
    # Formula: (1/sqrt(2pi)) * e^(-0)
    expected = 1.0 / math.sqrt(2.0 * math.pi)
    assert math.isclose(math_utils.norm_pdf(0.0), expected, rel_tol=1e-9)


def test_norm_pdf_symmetry():
    # The normal distribution is symmetric around 0.
    # PDF(x) == PDF(-x)
    x = 1.5
    val_pos = math_utils.norm_pdf(x)
    val_neg = math_utils.norm_pdf(-x)

    assert math.isclose(val_pos, val_neg, rel_tol=1e-9)


def test_norm_pdf_tails():
    # PDF should approach 0 for large x
    assert math.isclose(math_utils.norm_pdf(10.0), 0.0, abs_tol=1e-9)


# --- Cumulative Distribution Function (CDF) Tests ---


def test_norm_cdf_at_zero():
    # CDF at the mean (0) must be exactly 0.5
    assert math.isclose(math_utils.norm_cdf(0.0), 0.5, rel_tol=1e-9)


def test_norm_cdf_standard_sigmas():
    # 1 Sigma (approx 84.13%)
    assert math.isclose(math_utils.norm_cdf(1.0), 0.8413447, rel_tol=1e-5)

    # 2 Sigma (approx 97.72%)
    assert math.isclose(math_utils.norm_cdf(2.0), 0.9772498, rel_tol=1e-5)

    # 1.96 Sigma (Used for 95% CI -> 97.5% cumulative)
    assert math.isclose(math_utils.norm_cdf(1.96), 0.9750021, rel_tol=1e-5)


def test_norm_cdf_limits():
    # As x -> -infinity, CDF -> 0
    assert math.isclose(math_utils.norm_cdf(-10.0), 0.0, abs_tol=1e-9)

    # As x -> +infinity, CDF -> 1
    assert math.isclose(math_utils.norm_cdf(10.0), 1.0, abs_tol=1e-9)


def test_consistency_pdf_cdf():
    # The derivative of CDF is PDF.
    # CDF(x+h) - CDF(x-h) / 2h ~ PDF(x)
    x = 0.5
    h = 1e-5

    numeric_pdf = (math_utils.norm_cdf(x + h) - math_utils.norm_cdf(x - h)) / (2 * h)
    analytic_pdf = math_utils.norm_pdf(x)

    assert math.isclose(numeric_pdf, analytic_pdf, rel_tol=1e-5)
