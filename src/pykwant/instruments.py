from dataclasses import dataclass
from datetime import date
from typing import NewType

from toolz import pipe
from toolz.curried import filter, map

from pykwant import dates, rates

Money = NewType("Money", float)


@dataclass(frozen=True)
class CashFlow:
    """
    Docstring for CashFlow
    """

    amount: Money
    payment_date: date
    type: str = "coupon"  # "coupon", "principal", "fee"


@dataclass(frozen=True)
class FixedRateBond:
    """
    Docstring for FixedRateBond
    """

    face_value: Money
    coupon_rate: float
    start_date: date
    maturity_date: date
    frequency_months: int
    day_count: dates.DayCountConvention = dates.thirty_360
    calendar: dates.Calendar = dates.Calendar(holidays=frozenset())
    rolling: dates.RollingConvention = dates.modified_following


def generate_cash_flows(bond: FixedRateBond) -> list[CashFlow]:
    """
    Docstring for generate_cash_flows
    """
    payment_dates: list[date] = dates.generate_schedule(
        start=bond.start_date,
        end=bond.maturity_date,
        freq_month=bond.frequency_months,
        cal=bond.calendar,
        convention=bond.rolling,
    )

    flows = []

    prev_date: date = payment_dates[0]

    for curr_date in payment_dates[1:]:
        tau = bond.day_count(prev_date, curr_date)
        amt = bond.face_value * bond.coupon_rate * tau
        flows.append(CashFlow(amount=Money(amt), payment_date=curr_date, type="coupon"))
        prev_date = curr_date

    flows.append(
        CashFlow(
            amount=Money(bond.face_value),
            payment_date=payment_dates[-1],
            type="principal",
        )
    )

    return flows


def price_instrument(
    instrument: FixedRateBond,
    curve: rates.YieldCurveFn,
    valuation_date: date,
) -> Money:
    """
    Docstring for price_instrument
    """

    def _pv_flow(cf: CashFlow) -> float:
        df = curve(cf.payment_date)
        return float(cf.amount * df)

    def _is_future(cf: CashFlow) -> bool:
        return cf.payment_date > valuation_date

    npv = pipe(instrument, generate_cash_flows, filter(_is_future), map(_pv_flow), sum)

    return Money(npv)


def accrued_interest(bond: FixedRateBond, valuation_date: date) -> Money:
    """
    Docstring for accrued_interest
    """
    schedule = dates.generate_schedule(
        start=bond.start_date,
        end=bond.maturity_date,
        freq_month=bond.frequency_months,
        cal=bond.calendar,
        convention=bond.rolling,
    )
    prev = [d for d in schedule if d <= valuation_date][-1]
    tau_accrued = bond.day_count(prev, valuation_date)
    accrued = bond.face_value * bond.coupon_rate * tau_accrued
    return Money(accrued)


def clean_price(
    instrument: FixedRateBond, curve: rates.YieldCurveFn, valuation_date: date
) -> Money:
    """
    Docstring for clean_price
    """
    dirty = price_instrument(instrument, curve, valuation_date)
    accrued = accrued_interest(instrument, valuation_date)
    return Money(dirty - accrued)
