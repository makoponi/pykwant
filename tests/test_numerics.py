import math

from pykwant.numerics import (
    linear_interpolation,
    log_linear_interpolation,
    newton_solve,
    numerical_derivative,
)


def test_linear_interpolation() -> None:
    x_data = [0.0, 1.0, 2.0]
    y_data = [0.0, 2.0, 4.0]
    interp = linear_interpolation(x_data, y_data)

    assert math.isclose(interp(0.5), 1.0)
    assert math.isclose(interp(1.5), 3.0)

    assert math.isclose(interp(0.0), 0.0)
    assert math.isclose(interp(2.0), 4.0)

    assert math.isclose(interp(-1.0), 0.0)
    assert math.isclose(interp(3.0), 4.0)


def test_linear_interpolation_no_extrapolate() -> None:
    x_data = [0.0, 1.0]
    y_data = [0.0, 1.0]
    interp = linear_interpolation(x_data, y_data, extrapolate=False)

    assert math.isnan(interp(-0.1))
    assert math.isnan(interp(1.1))
    assert math.isclose(interp(0.5), 0.5)


def test_log_linear_interpolation() -> None:
    x_data = [0.0, 1.0, 2.0]
    y_data = [1.0, math.exp(1), math.exp(2)]
    interp = log_linear_interpolation(x_data, y_data)

    assert math.isclose(interp(0.5), math.exp(0.5))


def test_numerical_derivative() -> None:
    def f(x: float) -> float:
        return x**2

    df = numerical_derivative(f, h=1e-5)
    assert math.isclose(df(3.0), 6.0, rel_tol=1e-5)


def test_newton_solve() -> None:
    def f(x: float) -> float:
        return x**2 - 4

    root = newton_solve(f, target=0.0, guess=1.0)
    assert root is not None
    assert math.isclose(root, 2.0)

    root = newton_solve(f, target=0.0, guess=-1.0)
    assert root is not None
    assert math.isclose(root, -2.0)


def test_newton_solve_failure() -> None:
    def f(x: float) -> float:
        return x**2 - 4

    root = newton_solve(f, target=0.0, guess=0.0)
    assert root is None
