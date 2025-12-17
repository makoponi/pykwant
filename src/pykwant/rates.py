import math
from datetime import date
from typing import Callable, NewType

from pykwant import dates, numerics

Rate = NewType("Rate", float)
DiscountFactor = NewType("DiscountFactor", float)

YieldCurveFn = Callable[[date], DiscountFactor]


def create_curve_from_discount_factor(
    reference_date: date,
    dates_list: list[date],
    dfs_list: list[float],
    day_count: dates.DayCountConvention = dates.act_365,
    interpolator_factory: Callable[
        [list[float], list[float]], numerics.Interpolator
    ] = numerics.log_linear_interpolation,
) -> YieldCurveFn:
    """
    Docstring for create_curve_from_discount_factor
    """
    times = [day_count(reference_date, d) for d in dates_list]
    math_curve = interpolator_factory(times, dfs_list)

    def _curve(d: date) -> DiscountFactor:
        if d == reference_date:
            return DiscountFactor(1.0)
        t = day_count(reference_date, d)
        df_value = math_curve(t)
        return DiscountFactor(df_value)

    return _curve


def zero_rates(
    curve: YieldCurveFn,
    reference_date: date,
    target_date: date,
    day_count: dates.DayCountConvention = dates.act_365,
) -> Rate:
    """
    Docstring for zero_rates
    """
    if target_date == reference_date:
        return Rate(0.0)
    df = curve(target_date)
    t = day_count(reference_date, target_date)
    r = -math.log(df) / t
    return Rate(r)


def forward_rate(
    curve: YieldCurveFn,
    start_date: date,
    end_date: date,
    day_count: dates.DayCountConvention = dates.act_365,
) -> Rate:
    """
    Docstring for forward_rate
    """
    df_start = curve(start_date)
    df_end = curve(end_date)
    tau = day_count(start_date, end_date)

    if tau == 0:
        return Rate(0.0)

    fwd = (df_start / df_end - 1.0) / tau
    return Rate(fwd)


def present_value(amount: float, payment_date: date, curve: YieldCurveFn) -> float:
    """
    Docstring for present_value
    """
    return amount * curve(payment_date)


def compound_factor(rate: Rate, t: float, frequency: int = 0) -> float:
    """
    Docstring for compound_factor
    """
    if frequency == 0:
        return math.exp(rate * t)
    return float((1 + rate / frequency) ** (frequency * t))
