import math

from pykwant import math_utils


def test_norm_pdf_at_zero() -> None:
    expected = 1.0 / math.sqrt(2.0 * math.pi)
    assert math.isclose(math_utils.norm_pdf(0.0), expected, rel_tol=1e-9)


def test_norm_pdf_symmetry() -> None:
    x = 1.5
    assert math_utils.norm_pdf(x) == math_utils.norm_pdf(-x)


def test_norm_cdf_at_zero() -> None:
    assert math_utils.norm_cdf(0.0) == 0.5


def test_norm_cdf_limits() -> None:
    assert math.isclose(math_utils.norm_cdf(10.0), 1.0, rel_tol=1e-9)
    assert math.isclose(math_utils.norm_cdf(-10.0), 0.0, abs_tol=1e-9)


def test_norm_cdf_symmetry_property() -> None:
    x = 0.5
    assert math.isclose(
        math_utils.norm_cdf(x) + math_utils.norm_cdf(-x), 1.0, rel_tol=1e-9
    )


def test_norm_cdf_known_values() -> None:
    assert math.isclose(math_utils.norm_cdf(1.0), 0.841344746, rel_tol=1e-7)

    assert math.isclose(math_utils.norm_cdf(1.96), 0.975, rel_tol=1e-3)
