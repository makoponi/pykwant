"""
Dates Module
============

This module handles date arithmetic, calendar management, and day count conventions
following a functional programming paradigm.

It provides:
- Immutable `Calendar` data structures.
- Pure functions for business day adjustments (Rolling Conventions).
- Standard Day Count Conventions (ACT/365, 30/360, etc.).
- Schedule generation for financial instruments.

All functions in this module are pure: they do not modify their inputs and
always return new date objects or values.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from enum import IntEnum
from typing import Callable, TypeAlias


class Month(IntEnum):
    """Enumeration for months to improve code readability."""

    JANUARY = 1
    FEBRUARY = 2
    MARCH = 3
    APRIL = 4
    MAY = 5
    JUNE = 6
    JULY = 7
    AUGUST = 8
    SEPTEMBER = 9
    OCTOBER = 10
    NOVEMBER = 11
    DECEMBER = 12


class Weekday(IntEnum):
    """Enumeration for weekdays (ISO 8601 compatible: Monday=1, Sunday=7)."""

    MONDAY = 1
    TUESDAY = 2
    WEDNESDAY = 3
    THURSDAY = 4
    FRIDAY = 5
    SATURDAY = 6
    SUNDAY = 7


@dataclass(frozen=True)
class Calendar:
    """
    Immutable representation of a financial calendar.

    Attributes:
        holidays (frozenset[date]): A set of dates representing holidays.
                                    Using frozenset ensures O(1) lookup and immutability.
        weekends (tuple[int, ...]): A tuple of integers representing weekend days
                                    (e.g., (6, 7) for Sat/Sun). Defaults to Sat/Sun.
    """

    holidays: frozenset[date] = frozenset()
    weekends: tuple[int, ...] = (Weekday.SATURDAY, Weekday.SUNDAY)


def is_business_day(d: date, cal: Calendar) -> bool:
    """
    Checks if a given date is a valid business day.

    Args:
        d (date): The date to check.
        cal (Calendar): The calendar containing holidays and weekend definitions.

    Returns:
        bool: True if the date is not a weekend and not a holiday, False otherwise.
    """
    if d.isoweekday() in cal.weekends:
        return False
    if d in cal.holidays:
        return False
    return True


def _adjust(d: date, cal: Calendar, step: int = 1) -> date:
    """
    Internal pure helper to shift a date until a business day is found.

    Args:
        d (date): Starting date.
        cal (Calendar): The calendar.
        step (int): Direction to move (+1 for forward, -1 for backward).

    Returns:
        date: The first valid business day found.
    """
    current = d
    while not is_business_day(current, cal):
        current += timedelta(days=step)
    return current


def following(d: date, cal: Calendar) -> date:
    """
    'Following' rolling convention.

    If the date is a business day, returns it.
    Otherwise, returns the first business day after the date.

    Args:
        d (date): The original payment date.
        cal (Calendar): The calendar to check against.

    Returns:
        date: The adjusted business day.
    """
    return _adjust(d, cal, step=1)


def preceding(d: date, cal: Calendar) -> date:
    """
    'Preceding' rolling convention.

    If the date is a business day, returns it.
    Otherwise, returns the first business day before the date.

    Args:
        d (date): The original payment date.
        cal (Calendar): The calendar to check against.

    Returns:
        date: The adjusted business day.
    """
    return _adjust(d, cal, step=-1)


def modified_following(d: date, cal: Calendar) -> date:
    """
    'Modified Following' rolling convention.

    Moves to the next business day (Following). However, if that day falls
    in the next month, it moves to the previous business day (Preceding) instead.
    Commonly used for month-end payments to ensure they stay within the same month.

    Args:
        d (date): The original payment date.
        cal (Calendar): The calendar.

    Returns:
        date: The adjusted business day.
    """
    next_biz = following(d, cal)
    if next_biz.month != d.month:
        return preceding(d, cal)
    return next_biz


# Type alias for Rolling Conventions
RollingConvention: TypeAlias = Callable[[date, Calendar], date]


# --- Day Count Conventions ---


def act_365(start: date, end: date) -> float:
    """
    ACT/365 Fixed day count convention.

    Calculates the year fraction as the actual number of days between dates
    divided by a fixed 365.

    Args:
        start (date): The start date.
        end (date): The end date.

    Returns:
        float: The year fraction (e.g., 0.25 for ~3 months).
    """
    return (end - start).days / 365.0


def act_360(start: date, end: date) -> float:
    """
    ACT/360 day count convention.

    Calculates the year fraction as the actual number of days between dates
    divided by a fixed 360. Often used in money markets.

    Args:
        start (date): The start date.
        end (date): The end date.

    Returns:
        float: The year fraction.
    """
    return (end - start).days / 360.0


def thirty_360(start: date, end: date) -> float:
    """
    30/360 (Bond Basis) day count convention.

    Assumes every month has 30 days. Useful for standard corporate bonds.
    Implements the standard ISDA logic (30U/360).

    Args:
        start (date): The start date.
        end (date): The end date.

    Returns:
        float: The year fraction.
    """
    d1 = start.day
    d2 = end.day
    m1 = start.month
    m2 = end.month
    y1 = start.year
    y2 = end.year

    if d1 == 31:
        d1 = 30
    if d2 == 31 and d1 == 30:
        d2 = 30

    days = 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)
    return days / 360.0


# Type alias for Day Count functions
DayCountConvention: TypeAlias = Callable[[date, date], float]


# --- Schedule Generation ---


def generate_schedule(
    start: date,
    end: date,
    freq_month: int,
    cal: Calendar,
    convention: RollingConvention = modified_following,
) -> list[date]:
    """
    Generates a list of adjusted payment dates.

    This function creates a sequence of theoretical unadjusted dates by adding
    months to the start date, and then adjusts each one according to the
    provided rolling convention and calendar.

    Args:
        start (date): The start date of the schedule (e.g., Issue Date).
        end (date): The end date of the schedule (e.g., Maturity Date).
        freq_month (int): The frequency in months (e.g., 6 for semiannual).
        cal (Calendar): The calendar for holidays.
        convention (RollingConvention): The function to adjust non-business days.

    Returns:
        list[date]: A sorted list of valid business days representing the schedule.
                    Does not include the start date, includes the end date.
    """
    dates_list = []
    current = start

    # We generate dates until we reach or exceed the end date
    # Note: A robust implementation would handle month-end adjustments (e.g. 28th Feb -> 30th)
    # more explicitly. Here we rely on simple month addition logic.
    while True:
        # Simple month addition logic
        y = current.year + (current.month + freq_month - 1) // 12
        m = (current.month + freq_month - 1) % 12 + 1

        # Handle invalid days (e.g. Feb 30)
        # In a real library, this logic might be more sophisticated (e.g. moving to last day)
        try:
            current = date(y, m, start.day)
        except ValueError:
            # Fallback to last day of month if start.day doesn't exist (e.g. 31st -> 30th/28th)
            # This is a simplification; robust date math is complex.
            if m == 2:
                current = date(
                    y, m, 28
                )  # Leap year check omitted for brevity in this snippet
            else:
                current = date(y, m, 30)

        if current > end:
            break

        adjusted_date = convention(current, cal)
        dates_list.append(adjusted_date)

    # Ensure the maturity date is included and adjusted properly
    # Often the last generated date IS the maturity, but sometimes there's a stub.
    # For this simple implementation, we ensure the specific 'end' is added/adjusted if not present.
    adjusted_end = convention(end, cal)
    if not dates_list or dates_list[-1] != adjusted_end:
        dates_list.append(adjusted_end)

    return dates_list
