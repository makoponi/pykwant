"""
Instruments Module
==================

This module defines the financial instruments supported by the library.
It strictly follows the Functional Programming paradigm by separating:
1. **State**: Immutable `dataclasses` (e.g., `FixedRateBond`, `EuropeanOption`).
2. **Behavior**: Pure functions that operate on these structures (e.g., `price_instrument`).

Key Concepts:
- **Money**: A semantic alias for float to indicate monetary values.
- **CashFlow**: The atomic unit of valuation.
"""

from dataclasses import dataclass
from datetime import date
from typing import List, NewType, Union

from pykwant import dates, rates

# Semantic alias for monetary values (currently implemented as float)
# Using NewType creates a distinct type for static type checkers.
Money = NewType("Money", float)


@dataclass(frozen=True)
class CashFlow:
    """
    Represents a single atomic cash flow.

    Attributes:
        amount (Money): The monetary value of the flow.
        payment_date (date): The date when the payment occurs.
        type (str): The nature of the flow (e.g., 'principal', 'coupon', 'fee').
    """

    amount: Money
    payment_date: date
    type: str = "coupon"


@dataclass(frozen=True)
class FixedRateBond:
    """
    Immutable representation of a Fixed Rate Bond.

    Attributes:
        face_value (Money): The notional amount.
        coupon_rate (float): The annual coupon rate (decimal).
        start_date (date): The issue date.
        maturity_date (date): The final repayment date.
        frequency_months (int): Frequency of coupons in months.
        day_count (dates.DayCountConvention): Day count function.
        calendar (dates.Calendar): Calendar for rolling.
    """

    face_value: Money
    coupon_rate: float
    start_date: date
    maturity_date: date
    frequency_months: int
    day_count: dates.DayCountConvention
    calendar: dates.Calendar


@dataclass(frozen=True)
class EuropeanOption:
    """
    Immutable representation of a Vanilla European Option.
    Exercisable only at expiry.
    """

    asset_name: str
    strike: Money
    expiry_date: date
    call_put: str = "call"


@dataclass(frozen=True)
class AmericanOption:
    """
    Immutable representation of a Vanilla American Option.
    Exercisable at any time up to expiry.

    Attributes:
        asset_name (str): Identifier of the underlying.
        strike (Money): The strike price.
        expiry_date (date): The expiration date.
        call_put (str): "call" or "put".
    """

    asset_name: str
    strike: Money
    expiry_date: date
    call_put: str = "call"


# Definition of a Generic Instrument Type for type hinting
Instrument = Union[FixedRateBond, EuropeanOption, AmericanOption]


def generate_cash_flows(bond: FixedRateBond) -> List[CashFlow]:
    """
    Generates the deterministic schedule of cash flows for a Fixed Rate Bond.
    """
    flows = []

    # Generate unadjusted schedule dates
    schedule = dates.generate_schedule(
        start=bond.start_date,
        end=bond.maturity_date,
        freq_month=bond.frequency_months,
        cal=bond.calendar,
        convention=dates.modified_following,
    )

    current_start = bond.start_date

    # Calculate Coupons
    for pay_date in schedule:
        tau = bond.day_count(current_start, pay_date)
        amount = Money(bond.face_value * bond.coupon_rate * tau)
        flows.append(CashFlow(amount, pay_date, "coupon"))
        current_start = pay_date

    # Add Principal Repayment at maturity
    last_date = schedule[-1] if schedule else bond.maturity_date
    flows.append(CashFlow(bond.face_value, last_date, "principal"))

    return flows


def accrued_interest(bond: FixedRateBond, valuation_date: date) -> float:
    """
    Calculates the Accrued Interest for a bond at a given valuation date.
    """
    if valuation_date < bond.start_date:
        return 0.0

    schedule = dates.generate_schedule(
        start=bond.start_date,
        end=bond.maturity_date,
        freq_month=bond.frequency_months,
        cal=bond.calendar,
    )

    prev_date = bond.start_date
    for d in schedule:
        if d > valuation_date:
            break
        prev_date = d

    if prev_date == valuation_date:
        return 0.0

    tau = bond.day_count(prev_date, valuation_date)
    return bond.face_value * bond.coupon_rate * tau


def price_instrument(
    instrument: Instrument, curve: rates.YieldCurveFn, valuation_date: date
) -> float:
    """
    Calculates the Dirty Price (NPV) of an instrument.
    Dispatches to specific modules for Options to avoiding circular imports if possible,
    or raises error if external modules must be used directly.
    """
    if isinstance(instrument, FixedRateBond):
        flows = generate_cash_flows(instrument)
        npv = 0.0
        for cf in flows:
            if cf.payment_date > valuation_date:
                npv += rates.present_value(cf.amount, cf.payment_date, curve)
        return npv

    if isinstance(instrument, (EuropeanOption, AmericanOption)):
        raise NotImplementedError(
            "Option pricing is handled in 'equity' (European) or 'trees' (American) modules."
        )

    raise NotImplementedError(f"Pricing not implemented for {type(instrument)}")


def clean_price(bond: FixedRateBond, curve: rates.YieldCurveFn, valuation_date: date) -> float:
    """
    Calculates the Clean Price of a bond.
    """
    dirty = price_instrument(bond, curve, valuation_date)
    accrued = accrued_interest(bond, valuation_date)
    return dirty - accrued
