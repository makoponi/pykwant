"""
Models Module
=============

This module implements stochastic interest rate models, primarily the **Hull-White 1-Factor Model**.

The Hull-White model extends the Vasicek model to fit the initial market term structure
of interest rates perfectly by making the mean-reversion level time-dependent.

Dynamics:
$$ dr_t = [\\theta(t) - a r_t]dt + \\sigma dW_t $$

Key Features:
- **Analytical ZCB Pricing**: Closed-form solution for Bond prices $P(t, T)$.
- **Short Rate Simulation**: Monte Carlo generation of $r_t$ paths consistent 
                             with the initial curve.
"""

import math
import random
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable, List, Tuple

from pykwant import rates

# Type alias for a path of short rates
RatePath = List[float]


@dataclass(frozen=True)
class HullWhiteModel:
    """
    Parameters for the Hull-White 1-Factor Model.

    Attributes:
        mean_reversion (float): The speed of mean reversion ($a$).
        volatility (float): The volatility of the short rate ($\\sigma$).
    """

    mean_reversion: float
    volatility: float


def create_time_domain_curve(
    curve: rates.YieldCurveFn, reference_date: date
) -> Callable[[float], float]:
    """
    Wraps a standard date-based YieldCurveFn into a time-based function `f(t) -> DF`.

    Models often work with continuous time $t$ (years), while market curves works with dates.
    This helper bridges the gap using ACT/365 convention.

    Args:
        curve (YieldCurveFn): The date-based yield curve.
        reference_date (date): The anchor date corresponding to $t=0$.

    Returns:
        Callable[[float], float]: A function taking time $t$ and returning a Discount Factor.
    """

    def _time_curve(t: float) -> float:
        # Convert time (years) to date delta assuming ACT/365
        days = round(t * 365.0)
        target_date = reference_date + timedelta(days=days)
        return curve(target_date)

    return _time_curve


def _calculate_A_B(model: HullWhiteModel, t: float, T: float) -> Tuple[float, float]:
    """
    Calculates the A(t, T) and B(t, T) coefficients for the affine term structure.

    $$ P(t, T) = A(t, T) e^{-B(t, T) r_t} $$
    """
    a = model.mean_reversion
    tau = T - t

    if tau == 0:
        return 1.0, 0.0

    # B(t, T) = (1 - exp(-a * tau)) / a
    if a == 0:
        B = tau
    else:
        B = (1.0 - math.exp(-a * tau)) / a

    # A(t, T) formula is complex for Hull-White because it depends on the
    # initial curve. In practice, we use the relationship:
    # P(t, T) = P(0, T) / P(0, t) * exp(...)
    # This helper returns the 'model specific' part often denoted in textbooks,
    # but for calibration, we usually embed the P(0,T) ratio directly in the pricer.
    # Here we return the B coefficient which is standard.
    # We will handle the full P(t,T) logic in `zcb_price`.

    return 0.0, B  # A is context dependent, B is structural


def zcb_price(
    model: HullWhiteModel, curve: Callable[[float], float], r_t: float, t: float, T: float
) -> float:
    r"""
    Calculates the price of a Zero Coupon Bond $P(t, T)$ under the Hull-White model.

    Formula:
    $$ P(t, T) = \\frac{P(0, T)}{P(0, t)} \\exp\\left( B(t, T) f(0, t) - \\frac{\\sigma^2}{4a} (1 - e^{-2at}) B(t, T)^2 - B(t, T) r_t \\right) $$

    Wait, a more stable form uses the forward rate adjustment:
    $$ P(t, T) = A(t, T) e^{-B(t, T) r_t} $$
    Where $A(t, T)$ ensures $P(0, T)$ matches the market curve.

    Simplified formula fitting initial curve:
    $$ P(t, T) = \\frac{P^M(0, T)}{P^M(0, t)} \\exp\\left( B(t, T) f^M(0, t) - \\frac{\\sigma^2}{4a}(1-e^{-2at})B(t,T)^2 - B(t,T)r_t \\right) $$

    *Correction*: The standard formula is:
    $$ P(t, T) = \\frac{P(0, T)}{P(0, t)} \\exp\\left( -B(t, T) r_t^* - \\frac{1}{2} V(t, T) \\right) $$
    where $r_t^* = r_t - f(0, t)$? No, let's use the explicit shift formulation.

    Let's implement the standard textbook formula (Brigo & Mercurio):
    $$ P(t, T) = A(t, T) e^{-B(t, T) r_t} $$
    $$ \ln A(t, T) = \ln \frac{P(0, T)}{P(0, t)} + B(t, T) f(0, t) - \frac{\sigma^2}{4a} (1 - e^{-2at}) B(t, T)^2 $$

    Args:
        model (HullWhiteModel): Model parameters.
        curve (Callable[[float], float]): Initial yield curve `f(t) -> DF`.
                                          Use `create_time_domain_curve` to adapt a standard curve.
        r_t (float): The short rate at time t.
        t (float): Current time in years.
        T (float): Maturity time in years.

    Returns:
        float: The ZCB price.
    """  # noqa: E501
    if t >= T:
        return 1.0

    a = model.mean_reversion
    sigma = model.volatility

    # 1. Market Discount Factors
    # We use the time-domain curve wrapper passed by the user
    P_0_T = curve(T)
    P_0_t = curve(t)

    # 2. Instantaneous Forward Rate f(0, t)
    # f(0, t) = - d/dt ln P(0, t)
    # We compute this numerically using the curve
    h = 1e-4
    f_0_t = -(math.log(curve(t + h)) - math.log(curve(t - h))) / (2 * h)

    # 3. B(t, T)
    _, B = _calculate_A_B(model, t, T)

    # 4. A(t, T) components
    # drift correction term: sigma^2 / (4a) * (1 - exp(-2at)) * B^2
    drift_correction = (sigma**2 / (4 * a)) * (1.0 - math.exp(-2 * a * t)) * (B**2)

    ln_A = math.log(P_0_T / P_0_t) + B * f_0_t - drift_correction

    return math.exp(ln_A - B * r_t)


def simulate_short_rate_paths(
    model: HullWhiteModel,
    curve: Callable[[float], float],
    time_horizon: float,
    steps: int,
    num_paths: int,
    seed: int | None = None,
) -> List[RatePath]:
    r"""
    Simulates paths for the Short Rate $r_t$ under Hull-White dynamics.

    It decomposes $r_t$ into a deterministic shift $\\alpha(t)$ and a stochastic
    Ornstein-Uhlenbeck process $x_t$.

    $$ r_t = x_t + \\alpha(t) $$
    $$ dx_t = -a x_t dt + \\sigma dW_t $$

    $\\alpha(t)$ is calibrated to the initial yield curve.

    Args:
        model (HullWhiteModel): Model parameters (a, sigma).
        curve (Callable[[float], float]): Initial curve `f(t) -> DF`.
        time_horizon (float): Simulation horizon in years.
        steps (int): Number of time steps.
        num_paths (int): Number of paths.
        seed (int, optional): Random seed.

    Returns:
        List[RatePath]: List of short rate paths.
    """
    if seed is not None:
        random.seed(seed)

    a = model.mean_reversion
    sigma = model.volatility
    dt = time_horizon / steps

    # Pre-calculate alpha(t) for each time step
    # alpha(t) = f(0, t) + sigma^2/(2a^2) * (1 - exp(-at))^2
    alphas = []
    times = [i * dt for i in range(steps + 1)]

    for t in times:
        # Instantaneous forward f(0, t)
        h = 1e-4
        # Handle t=0 edge case for numerical derivative
        t_plus = t + h
        t_minus = max(0.0, t - h)
        dt_actual = t_plus - t_minus

        f_0_t = -(math.log(curve(t_plus)) - math.log(curve(t_minus))) / dt_actual

        convexity_adj = (sigma**2 / (2 * a**2)) * (1.0 - math.exp(-a * t)) ** 2
        alphas.append(f_0_t + convexity_adj)

    # Simulate x_t (OU Process)
    # x_{t+1} = x_t * exp(-a*dt) + sigma * sqrt((1-exp(-2a*dt))/(2a)) * Z
    # Exact discretization is better than Euler for mean reversion

    decay = math.exp(-a * dt)
    # Variance of the shock over dt for OU process
    variance_step = (sigma**2 / (2 * a)) * (1.0 - math.exp(-2 * a * dt))
    std_step = math.sqrt(variance_step)

    all_paths = []

    for _ in range(num_paths):
        x = 0.0  # x_0 is always 0
        path = []

        for i in range(steps + 1):
            # r_t = x_t + alpha(t)
            r_t = x + alphas[i]
            path.append(r_t)

            if i < steps:
                z = random.gauss(0.0, 1.0)
                x = x * decay + std_step * z

        all_paths.append(path)

    return all_paths
