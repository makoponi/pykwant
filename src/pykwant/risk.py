import math
from datetime import date
from typing import Callable

import pykwant.instruments as instruments
import pykwant.rates as rates

PricerFn = Callable[
    [instruments.FixedRateBond, rates.YieldCurveFn, date], instruments.Money
]


def shift_curve(
    original_curve: rates.YieldCurveFn, delta: float, ref_date: date
) -> rates.YieldCurveFn:
    """
    Docstring for shift_curve
    """

    def _shifted_curve(d: date) -> rates.DiscountFactor:
        df_old = original_curve(d)
        t = (d - ref_date).days / 365
        adjustment = math.exp(-delta * t)
        return rates.DiscountFactor(df_old * adjustment)

    return _shifted_curve


def calculate_risk_metrics(
    instruments: instruments.FixedRateBond,
    curve: rates.YieldCurveFn,
    valuation_date: date,
    pricer: PricerFn = instruments.price_instrument,
    bump: float = 1e-4,  # 1 basis point
) -> dict[str, float]:
    """
    Docstring for calculate_risk_metrics
    """
    p0 = pricer(instruments, curve, valuation_date)

    if float(p0) == 0:
        return {"price": 0.0, "duration": 0.0, "convexity": 0.0, "dv01": 0.0}

    curve_up = shift_curve(curve, bump, valuation_date)
    curve_down = shift_curve(curve, -bump, valuation_date)

    p_up = pricer(instruments, curve_up, valuation_date)
    p_down = pricer(instruments, curve_down, valuation_date)

    duration = -(p_up - p_down) / (2 * p0 * bump)

    convexity = (p_up - 2 * p0 + p_down) / (p0 * bump**2)

    dv01 = duration * p0 * 0.0001

    return {"price": p0, "duration": duration, "convexity": convexity, "dv01": dv01}
