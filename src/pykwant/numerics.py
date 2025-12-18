"""
Numerics Module
===============

This module provides fundamental numerical algorithms implemented using a
functional programming paradigm.

It includes functionality for:
- **Interpolation**: Creating interpolation functions (Linear, Log-Linear) as closures.
- **Differentiation**: Computing numerical derivatives via higher-order functions.
- **Root Finding**: Solving equations (e.g., $f(x) = y$) using the Newton-Raphson method.

Unlike traditional vector-oriented scientific libraries (like NumPy), this module
focuses on creating composable *Callables*.
"""

import math
from typing import Callable, Optional

# Define a type alias for scalar functions f: float -> float
ScalarFunction = Callable[[float], float]


def linear_interpolation(
    x_data: list[float], y_data: list[float], extrapolate: bool = True
) -> ScalarFunction:
    """
    Constructs a linear interpolation function from the provided data.

    Returns a 'closure' (inner function) that captures the data and computes
    the interpolated value $y$ for any given $x$.

    Args:
        x_data (list[float]): Sorted list of x-coordinates (abscissas).
        y_data (list[float]): List of corresponding y-coordinates (ordinates).
        extrapolate (bool, optional): If True, allows linear extrapolation outside
            the domain [min(x), max(x)]. If False, returns `nan` for values
            outside the domain. Defaults to True.

    Returns:
        ScalarFunction: A function `f(x) -> y` that performs the interpolation.

    Raises:
        ValueError: If x_data and y_data have different lengths.
    """
    if len(x_data) != len(y_data):
        raise ValueError("x_data and y_data must have the same length.")

    # We assume x_data is sorted for performance.
    # In production, a check or sort could be added (but might break alignment with y).

    # Capture data in the closure (tuple for immutability and slight perf gain)
    xs = tuple(x_data)
    ys = tuple(y_data)
    n = len(xs)

    def _interpolator(x: float) -> float:
        # Handle extrapolation / boundaries
        if x <= xs[0]:
            if not extrapolate and x < xs[0]:
                return float("nan")
            # Slope of the first segment
            slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
            return ys[0] + slope * (x - xs[0])

        if x >= xs[-1]:
            if not extrapolate and x > xs[-1]:
                return float("nan")
            # Slope of the last segment
            slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
            return ys[-1] + slope * (x - xs[-1])

        # Binary or linear search for the segment
        # Using linear search here for simplicity (can be optimized with bisect)
        i = 0
        while i < n - 1 and xs[i + 1] < x:
            i += 1

        # Interpolate between i and i+1
        x0, x1 = xs[i], xs[i + 1]
        y0, y1 = ys[i], ys[i + 1]

        slope = (y1 - y0) / (x1 - x0)
        return y0 + slope * (x - x0)

    return _interpolator


def log_linear_interpolation(
    x_data: list[float], y_data: list[float], extrapolate: bool = True
) -> ScalarFunction:
    """
    Constructs a Log-Linear interpolation function.

    Transforms $y$ into $\ln(y)$, performs linear interpolation in the logarithmic
    space, and then converts the result back via the exponential function.
    Ideal for Discount Factors ($DF = e^{-rt}$).

    Args:
        x_data (list[float]): Sorted list of x-coordinates.
        y_data (list[float]): List of y-coordinates (must be > 0).
        extrapolate (bool, optional): Enable/Disable extrapolation. Defaults to True.

    Returns:
        ScalarFunction: A function `f(x) -> y`.
    """
    # Transform to logarithmic space
    log_ys = [math.log(y) for y in y_data]

    # Create the linear interpolator on the transformed space
    lin_interp = linear_interpolation(x_data, log_ys, extrapolate)

    def _log_interpolator(x: float) -> float:
        val = lin_interp(x)
        if math.isnan(val):
            return val
        return math.exp(val)

    return _log_interpolator


def numerical_derivative(func: ScalarFunction, h: float = 1e-5) -> ScalarFunction:
    """
    Returns the first numerical derivative of a given function.

    Uses the central difference method:
    $$ f'(x) \approx \frac{f(x+h) - f(x-h)}{2h} $$

    Args:
        func (ScalarFunction): The function f(x) to differentiate.
        h (float, optional): The step size for differentiation. Defaults to 1e-5.

    Returns:
        ScalarFunction: A new function f'(x) representing the approximate derivative.
    """

    def _derivative(x: float) -> float:
        return (func(x + h) - func(x - h)) / (2 * h)

    return _derivative


def newton_solve(
    func: ScalarFunction,
    target: float,
    guess: float,
    tol: float = 1e-7,
    max_iter: int = 100,
) -> Optional[float]:
    """
    Finds the root $x$ such that $func(x) = target$ using the Newton-Raphson method.

    Calculates the derivative (gradient) on the fly using `numerical_derivative`,
    so the user does not need to provide an explicit gradient function.

    Args:
        func (ScalarFunction): The objective function f(x).
        target (float): The target y value that f(x) should reach.
        guess (float): Initial guess for x.
        tol (float, optional): Tolerance for convergence. Defaults to 1e-7.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.

    Returns:
        Optional[float]: The found value x, or None if the algorithm fails to converge.
    """

    # Objective function: g(x) = f(x) - target = 0
    def objective(x: float) -> float:
        return func(x) - target

    # Automatically obtain the derivative
    grad = numerical_derivative(objective)

    x_curr = guess
    for _ in range(max_iter):
        y_val = objective(x_curr)

        if abs(y_val) < tol:
            return x_curr

        slope = grad(x_curr)
        if slope == 0:
            return None  # Zero gradient, cannot proceed

        # Newton step: x_new = x - f(x)/f'(x)
        x_curr = x_curr - y_val / slope

    return None
