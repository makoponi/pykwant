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

    This structure holds only the static data describing the bond contract.
    It does not contain pricing logic or market state.

    Attributes:
        face_value (Money): The notional amount (e.g., 100.0).
        coupon_rate (float): The annual coupon rate in decimal format (e.g., 0.05 for 5%).
        start_date (date): The issue date or interest accrual start date.
        maturity_date (date): The final repayment date.
        frequency_months (int): Frequency of coupons in months (e.g., 12=Annual, 6=Semi).
        day_count (dates.DayCountConvention): Function to calculate year fractions.
        calendar (dates.Calendar): Calendar used for rolling business days.
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

    Attributes:
        asset_name (str): Identifier of the underlying asset (e.g., 'AAPL', 'EURUSD').
        strike (Money): The strike price of the option.
        expiry_date (date): The expiration date.
        call_put (str): Option type, either "call" or "put". Defaults to "call".
    """

    asset_name: str
    strike: Money
    expiry_date: date
    call_put: str = "call"


# Definition of a Generic Instrument Type for type hinting
Instrument = Union[FixedRateBond, EuropeanOption]


def generate_cash_flows(bond: FixedRateBond) -> List[CashFlow]:
    """
    Generates the deterministic schedule of cash flows for a Fixed Rate Bond.

    This pure function calculates coupon dates and amounts based on the bond definition.
    It handles date rolling (business day adjustments) internally.

    Args:
        bond (FixedRateBond): The bond definition.

    Returns:
        List[CashFlow]: A sorted list of CashFlow objects, including coupons and
        principal repayment.
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
        # Calculate year fraction for the coupon period
        # Note: In production, the start date of the period might also need adjustment
        # depending on the specific convention (Adjusted vs Unadjusted calculation).
        # Here we assume Unadjusted period dates for fraction calculation.
        tau = bond.day_count(current_start, pay_date)

        # Coupon Amount = Face * Rate * Time
        amount = Money(bond.face_value * bond.coupon_rate * tau)

        flows.append(CashFlow(amount, pay_date, "coupon"))
        current_start = pay_date

    # Add Principal Repayment at maturity
    # Note: Maturity flow payment date usually matches the last coupon date
    last_date = schedule[-1] if schedule else bond.maturity_date
    flows.append(CashFlow(bond.face_value, last_date, "principal"))

    return flows


def accrued_interest(bond: FixedRateBond, valuation_date: date) -> float:
    """
    Calculates the Accrued Interest for a bond at a given valuation date.

    Accrued interest is the interest that has accumulated since the last
    coupon payment but has not yet been paid.

    Args:
        bond (FixedRateBond): The bond instrument.
        valuation_date (date): The date for calculation.

    Returns:
        float: The accrued interest amount. Returns 0.0 if valuation_date is
        before start_date or exactly on a coupon date.
    """
    if valuation_date < bond.start_date:
        return 0.0

    # Generate schedule to find the previous coupon date
    # Optimization: For regular bonds, we could calculate this analytically
    # without generating the full list, but this is safer and cleaner.
    schedule = dates.generate_schedule(
        start=bond.start_date,
        end=bond.maturity_date,
        freq_month=bond.frequency_months,
        cal=bond.calendar,
    )

    # Find the start of the current period
    # Default to bond start date
    prev_date = bond.start_date

    for d in schedule:
        if d > valuation_date:
            break
        prev_date = d

    # If valuation date is exactly on a payment date, accrued is usually 0 (or handled by the flow)
    if prev_date == valuation_date:
        return 0.0

    tau = bond.day_count(prev_date, valuation_date)
    return bond.face_value * bond.coupon_rate * tau


def price_instrument(
    instrument: Instrument, curve: rates.YieldCurveFn, valuation_date: date
) -> float:
    """
    Calculates the Dirty Price (NPV) of an instrument.

    This function acts as a dispatcher. It determines the type of instrument
    and applies the appropriate pricing logic (Discounted Cash Flow for bonds).

    Args:
        instrument (Instrument): The financial instrument (Bond, Option, etc.).
        curve (YieldCurveFn): The market yield curve for discounting.
        valuation_date (date): The date on which to price the instrument.

    Returns:
        float: The Present Value (Dirty Price) of the instrument.

    Raises:
        NotImplementedError: If the instrument type is not supported.
    """
    if isinstance(instrument, FixedRateBond):
        flows = generate_cash_flows(instrument)
        npv = 0.0
        for cf in flows:
            # Only discount future flows
            if cf.payment_date > valuation_date:
                npv += rates.present_value(cf.amount, cf.payment_date, curve)
        return npv

    if isinstance(instrument, EuropeanOption):
        # We assume the equity module will handle this, or we dispatch to it.
        # To avoid circular imports, we might keep specific option logic in equity.py
        # or implement a simple dispatcher here if logic is imported.
        # For now, we raise to indicate separation of concerns (handled in equity.py)
        raise NotImplementedError(
            "Option pricing is handled in the 'equity' module via 'black_scholes_price'."
        )

    raise NotImplementedError(f"Pricing not implemented for {type(instrument)}")


def clean_price(
    bond: FixedRateBond, curve: rates.YieldCurveFn, valuation_date: date
) -> float:
    """
    Calculates the Clean Price of a bond.

    Clean Price = Dirty Price (NPV) - Accrued Interest.
    This is the standard quoting convention for bonds in most markets.

    Args:
        bond (FixedRateBond): The bond instrument.
        curve (YieldCurveFn): The yield curve.
        valuation_date (date): The valuation date.

    Returns:
        float: The clean price.
    """
    dirty = price_instrument(bond, curve, valuation_date)
    accrued = accrued_interest(bond, valuation_date)
    return dirty - accrued
