import bisect
import math
from typing import Callable, Optional

Interpolator = Callable[[float], float]

ScalarFunction = Callable[[float], float]


def linear_interpolation(
    x_data: list[float], y_data: list[float], extrapolate: bool = True
) -> Interpolator:
    """
    Docstring for linear_interpolation
    """
    data: list[tuple[float, float]] = sorted(zip(x_data, y_data))
    xs, ys = zip(*data)

    def _interpolate(x: float) -> float:
        if x <= xs[0]:
            return ys[0] if extrapolate else float("nan")
        if x >= xs[-1]:
            return ys[-1] if extrapolate else float("nan")
        i: int = bisect.bisect_left(xs, x)
        x1, x2 = xs[i - 1], xs[i]
        y1, y2 = ys[i - 1], ys[i]
        slope: float = (y2 - y1) / (x2 - x1)
        return y1 + slope * (x - x1)  # type: ignore

    return _interpolate


def log_linear_interpolation(x_data: list[float], y_data: list[float]) -> Interpolator:
    """
    Docstring for log_linear_interpolation
    """
    log_ys = [math.log(y) for y in y_data]
    log_interp = linear_interpolation(x_data, log_ys)

    def _interpolate(x: float) -> float:
        return math.exp(log_interp(x))

    return _interpolate


def numerical_derivative(f: ScalarFunction, h: float = 1e-5) -> ScalarFunction:
    """
    Docstring for numerical_derivative
    """

    def _diff(x: float) -> float:
        return (f(x + h) - f(x - h)) / (2 * h)

    return _diff


def newton_solve(
    func: ScalarFunction,
    target: float,
    guess: float = 0.5,
    tol: float = 1e-7,
    max_iter: int = 100,
) -> Optional[float]:
    """
    Docstring for newton_solve
    """

    def g(x: float) -> float:
        return func(x) - target

    dg = numerical_derivative(g)
    x = guess
    for _ in range(max_iter):
        y = g(x)
        if abs(y) < tol:
            return x
        derivative = dg(x)
        if derivative == 0:
            return None
        x -= y / derivative
    return None
