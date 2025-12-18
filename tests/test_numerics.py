"""
Test suite for pykwant.numerics module.
"""

import math

import pytest

from pykwant import numerics

# --- 1. Linear Interpolation Tests ---

def test_linear_interpolation_basic():
    # y = 2x
    x_data = [1.0, 2.0, 3.0]
    y_data = [2.0, 4.0, 6.0]
    
    interp = numerics.linear_interpolation(x_data, y_data)
    
    # Exact points
    assert interp(1.0) == 2.0
    assert interp(2.0) == 4.0
    
    # Midpoint
    assert interp(1.5) == 3.0
    
    # Extrapolation
    assert interp(0.0) == 0.0  # Lower bound
    assert interp(4.0) == 8.0  # Upper bound

def test_linear_interpolation_no_extrapolation():
    x_data = [0.0, 10.0]
    y_data = [0.0, 100.0]
    
    interp = numerics.linear_interpolation(x_data, y_data, extrapolate=False)
    
    # Inside is fine
    assert interp(5.0) == 50.0
    
    # Outside returns NaN
    assert math.isnan(interp(-1.0))
    assert math.isnan(interp(11.0))

def test_linear_interpolation_errors():
    with pytest.raises(ValueError):
        numerics.linear_interpolation([1.0], [1.0, 2.0])

# --- 2. Log-Linear Interpolation Tests ---

def test_log_linear_interpolation():
    # Function: y = exp(x)
    # We provide points at x=0, 1, 2
    x_data = [0.0, 1.0, 2.0]
    y_data = [math.exp(0.0), math.exp(1.0), math.exp(2.0)]
    
    # Log-Linear interpolation of exp(x) should be exact (within float precision)
    # because ln(exp(x)) = x, which is linear.
    interp = numerics.log_linear_interpolation(x_data, y_data)
    
    # Test at x=0.5 -> exp(0.5) = 1.6487...
    expected = math.exp(0.5)
    result = interp(0.5)
    
    assert math.isclose(result, expected, rel_tol=1e-9)
    
    # Extrapolation: x=3.0 -> exp(3.0)
    assert math.isclose(interp(3.0), math.exp(3.0), rel_tol=1e-9)

def test_log_linear_returns_nan_no_extrap():
    x = [0.0, 1.0]
    y = [1.0, 2.718]
    interp = numerics.log_linear_interpolation(x, y, extrapolate=False)
    
    assert math.isnan(interp(2.0))

# --- 3. Numerical Differentiation Tests ---

def test_numerical_derivative_quadratic():
    # f(x) = x^2  => f'(x) = 2x
    def func(x):
        return x**2
    
    deriv = numerics.numerical_derivative(func, h=1e-5)
    
    # Check at x=3 -> should be 6
    assert math.isclose(deriv(3.0), 6.0, rel_tol=1e-5)
    
    # Check at x=0 -> should be 0
    assert math.isclose(deriv(0.0), 0.0, abs_tol=1e-5)

def test_numerical_derivative_cubic():
    # f(x) = x^3 => f'(x) = 3x^2
    def func(x):
        return x**3
    
    deriv = numerics.numerical_derivative(func)
    
    # Check at x=2 -> 3*(2^2) = 12
    assert math.isclose(deriv(2.0), 12.0, rel_tol=1e-5)

# --- 4. Newton Solve Tests ---

def test_newton_solve_basic():
    # Solve x^2 = 4
    # Roots: 2.0, -2.0
    def f(x):
        return x**2
    
    # Guess near positive root
    root_pos = numerics.newton_solve(f, target=4.0, guess=1.0)
    assert root_pos is not None
    assert math.isclose(root_pos, 2.0, rel_tol=1e-6)
    
    # Guess near negative root
    root_neg = numerics.newton_solve(f, target=4.0, guess=-1.0)
    assert root_neg is not None
    assert math.isclose(root_neg, -2.0, rel_tol=1e-6)

def test_newton_solve_no_convergence():
    # Solve exp(x) = 0 (impossible)
    # This should fail or run out of iterations
    def f(x):
        return math.exp(x)
        
    root = numerics.newton_solve(f, target=0.0, guess=0.0, max_iter=10)
    # It might drift to -infinity or return None if slope is too flat/handling
    # In our implementation, it likely runs out of iterations or returns None if grad is 0
    # For exp(x), gradient never zero, but it won't reach target.
    # Our implementation returns the last x if it doesn't converge? 
    # Checking implementation: returns None if not converged.
    assert root is None

def test_newton_solve_zero_gradient():
    # f(x) = x^2, target=-1 (impossible)
    # Derivative is 2x. If guess is 0, deriv is 0 -> fail.
    def f(x):
        return x**2
        
    root = numerics.newton_solve(f, target=-1.0, guess=0.0)
    assert root is None
