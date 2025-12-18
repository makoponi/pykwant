import math
from datetime import date

from pykwant import dates, instruments, math_utils, rates


def black_scholes_price(
    option: instruments.EuropeanOption,
    spot: float,
    volatility: float,
    curve: rates.YieldCurveFn,
    valuation_date: date,
) -> instruments.Money:
    """
    Docstring for black_scholes_price
    """
    if valuation_date > option.expiry_date:
        return instruments.Money(0.0)

    T = dates.act_365(valuation_date, option.expiry_date)
    K = float(option.strike)
    S = spot
    sigma = volatility

    df = curve(option.expiry_date)
    r = -math.log(df) / T if T > 0 else 0.0

    if T == 0:
        intrinsic = max(S - K, 0) if option.type == "call" else max(K - S, 0)
        return instruments.Money(intrinsic)

    if sigma == 0:
        intrinsic = max(S - K, 0) if option.type == "call" else max(K - S, 0)
        return instruments.Money(intrinsic)

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option.type == "call":
        price = S * math_utils.norm_cdf(d1) - K * math.exp(
            -r * T
        ) * math_utils.norm_cdf(d2)
    else:
        price = K * math.exp(-r * T) * math_utils.norm_cdf(
            -d2
        ) - S * math_utils.norm_cdf(-d1)

    return instruments.Money(price)
