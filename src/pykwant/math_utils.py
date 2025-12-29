"""
Math Utils Module
=================

This module provides essential mathematical utility functions for quantitative finance.

It is designed to be lightweight and purely Python-based, avoiding heavy dependencies
like `scipy` or `numpy` for core functionalities. Currently, it focuses on
statistical functions required for option pricing models (e.g., Black-Scholes).
"""

import math


def norm_pdf(x: float) -> float:
    r"""
    Calculates the Probability Density Function (PDF) of the Standard Normal Distribution.

    The Standard Normal Distribution has a mean ($\mu$) of 0 and a standard 
    deviation ($\sigma$) of 1.

    Formula:
    $$ \phi(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{1}{2}x^2} $$

    Args:
        x (float): The point at which to evaluate the density.

    Returns:
        float: The probability density at x.
    """
    return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)


def norm_cdf(x: float) -> float:
    r"""
    Calculates the Cumulative Distribution Function (CDF) of the Standard Normal Distribution.

    This represents the probability that a standard normal random variable $Z$ takes
    a value less than or equal to $x$ ($P(Z \le x)$).

    It uses the Error Function (`math.erf`), which is part of the Python standard library,
    ensuring high precision and performance without external C extensions.

    Formula:
    $$ \Phi(x) = \frac{1}{2} \left[ 1 + \text{erf}\left( \frac{x}{\sqrt{2}} \right) \right] $$

    Args:
        x (float): The upper bound of the integral.

    Returns:
        float: The cumulative probability (value between 0.0 and 1.0).
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
