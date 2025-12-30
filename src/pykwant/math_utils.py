"""
Math Utils Module
=================

This module provides essential mathematical and statistical utility functions
for quantitative finance, designed to be **Zero-Dependency** (Pure Python).

It includes:
1.  **Gaussian Functions**: PDF, CDF, and Inverse CDF (PPF) for the Standard Normal Distribution.
2.  **Descriptive Statistics**: Mean, Variance, Standard Deviation.
3.  **Relationships**: Covariance, Correlation.
4.  **Quantiles**: Linear interpolation for percentiles (useful for VaR).

The Inverse CDF (`norm_ppf`) implements the Beasley-Springer-Moro algorithm to achieve
high precision (~7 decimal places) without requiring external libraries like `scipy`.
"""

import math

# --- 1. Gaussian Functions (Normal Distribution) ---


def norm_pdf(x: float) -> float:
    r"""
    Calculates the Probability Density Function (PDF) of the Standard Normal Distribution.

    The Standard Normal Distribution has a mean ($\mu$) of 0 and a
    standard deviation ($\sigma$) of 1.

    Formula:
    $$ \phi(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{1}{2}x^2} $$

    Args:
        x (float): The point at which to evaluate the density.

    Returns:
        float: The probability density at x.
    """
    return math.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)


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


def norm_ppf(p: float) -> float:
    r"""
    Calculates the Percent Point Function (Inverse CDF) of the Standard Normal Distribution.

    Given a probability $p$, returns the value $z$ such that $\Phi(z) = p$.
    This function is critical for calculating Value at Risk (VaR) and for
    generating random numbers in Monte Carlo simulations.

    Implementation:
    It uses the **Beasley-Springer-Moro** rational approximation algorithm, which is
    standard in quantitative finance for high accuracy (typically ~7 decimal places)
    without heavy dependencies.

    Args:
        p (float): The cumulative probability. Must be in the range (0, 1).

    Returns:
        float: The z-score corresponding to the probability $p$.

    Raises:
        ValueError: If $p \le 0$ or $p \ge 1$.
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError("Probability p must be between 0 and 1 exclusive.")

    # Constants for the Beasley-Springer-Moro approximation
    a0, a1, a2, a3 = 2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637
    b0, b1, b2, b3 = -8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833
    c0, c1, c2, c3 = 0.33747548227, 0.97616901909, 0.16079797149, 0.02764388103
    c4, c5, c6, c7, c8 = 0.00384057293, 0.00039518965, 0.00003217678, 0.00000028881, 0.00000039603

    y = p - 0.5
    if abs(y) < 0.42:
        r = y * y
        return (
            y
            * (((a3 * r + a2) * r + a1) * r + a0)
            / ((((b3 * r + b2) * r + b1) * r + b0) * r + 1.0)
        )
    else:
        if y > 0:
            r = 1.0 - p
        else:
            r = p

        r = math.log(-math.log(r))
        z = c0 + r * (c1 + r * (c2 + r * (c3 + r * (c4 + r * (c5 + r * (c6 + r * (c7 + r * c8)))))))

        if y < 0:
            return -z
        else:
            return z


# --- 2. Descriptive Statistics ---


def mean(data: list[float]) -> float:
    """
    Calculates the arithmetic mean of a dataset.

    Args:
        data (list[float]): A list of numerical values.

    Returns:
        float: The mean value.

    Raises:
        ValueError: If the input list is empty.
    """
    if not data:
        raise ValueError("Cannot calculate mean of empty list")
    return sum(data) / len(data)


def variance(data: list[float], is_sample: bool = True) -> float:
    """
    Calculates the variance of a dataset.

    Args:
        data (list[float]): A list of numerical values.
        is_sample (bool, optional): If True, calculates Sample Variance (divides by N-1).
            If False, calculates Population Variance (divides by N). Defaults to True.

    Returns:
        float: The variance value.

    Raises:
        ValueError: If data has fewer than 2 elements.
    """
    n = len(data)
    if n < 2:
        raise ValueError("Variance requires at least two data points")

    mu = mean(data)
    sum_sq_diff = sum((x - mu) ** 2 for x in data)

    divisor = (n - 1) if is_sample else n
    return sum_sq_diff / divisor


def std_dev(data: list[float], is_sample: bool = True) -> float:
    """
    Calculates the standard deviation of a dataset.

    Standard Deviation is the square root of the variance.

    Args:
        data (list[float]): A list of numerical values.
        is_sample (bool, optional): If True, calculates Sample StdDev (N-1). Defaults to True.

    Returns:
        float: The standard deviation value.
    """
    return math.sqrt(variance(data, is_sample))


# --- 3. Relationships & Advanced Stats ---


def covariance(x: list[float], y: list[float], is_sample: bool = True) -> float:
    """
    Calculates the covariance between two datasets X and Y.

    Covariance measures the joint variability of two random variables.
    Positive covariance indicates that the variables tend to move in the same direction.

    Args:
        x (list[float]): The first dataset.
        y (list[float]): The second dataset.
        is_sample (bool, optional): If True, uses Bessel's correction (N-1). Defaults to True.

    Returns:
        float: The covariance value.

    Raises:
        ValueError: If datasets have different lengths or fewer than 2 points.
    """
    n = len(x)
    if n != len(y):
        raise ValueError("Lists must have the same length")
    if n < 2:
        raise ValueError("Covariance requires at least two data points")

    mu_x = mean(x)
    mu_y = mean(y)

    sum_product = sum((xi - mu_x) * (yi - mu_y) for xi, yi in zip(x, y, strict=True))

    divisor = (n - 1) if is_sample else n
    return sum_product / divisor


def correlation(x: list[float], y: list[float]) -> float:
    """
    Calculates the Pearson correlation coefficient (rho).

    It is the normalized covariance, with values between -1.0 and +1.0.

    Args:
        x (list[float]): The first dataset.
        y (list[float]): The second dataset.

    Returns:
        float: The correlation coefficient. Returns 0.0 if one of the series has no variation.
    """
    cov = covariance(x, y, is_sample=True)
    sx = std_dev(x, is_sample=True)
    sy = std_dev(y, is_sample=True)

    if sx == 0 or sy == 0:
        return 0.0  # Safe default if variation is zero

    return cov / (sx * sy)


def percentile(data: list[float], p: float, sorted_data: bool = False) -> float:
    """
    Calculates the p-th percentile (0 <= p <= 1) using linear interpolation.

    This is equivalent to the 'inclusive' method used in Excel or `numpy.percentile`.
    It is useful for calculating Historical Value at Risk (VaR).

    Args:
        data (list[float]): The input dataset.
        p (float): The percentile rank (e.g., 0.95 for 95th percentile).
        sorted_data (bool, optional): Optimization flag. If True, assumes the input
            list is already sorted. Defaults to False.

    Returns:
        float: The interpolated value at the p-th percentile.

    Raises:
        ValueError: If p is outside [0, 1] or data is empty.
    """
    if not data:
        raise ValueError("Cannot calculate percentile of empty list")
    if not (0 <= p <= 1):
        raise ValueError("Percentile p must be between 0 and 1")

    # Sort if not already sorted (Pure Python sort is Timsort: O(N log N))
    dataset = data if sorted_data else sorted(data)
    n = len(dataset)

    # Calculate virtual index
    k = (n - 1) * p
    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return dataset[int(k)]

    # Linear interpolation between indices f and c
    d0 = dataset[int(f)]
    d1 = dataset[int(c)]
    fraction = k - f

    return d0 + (d1 - d0) * fraction
