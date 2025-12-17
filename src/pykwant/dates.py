from dataclasses import dataclass
from datetime import date, timedelta
from itertools import count, takewhile
from typing import Callable, FrozenSet, Generator

from dateutil.relativedelta import relativedelta

MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY = range(7)


@dataclass(frozen=True)
class Calendar:
    """
    Docstring for Calendar
    """

    holidays: FrozenSet[date]
    weekends: tuple[int, int] = (SATURDAY, SUNDAY)


def is_business_day(d: date, cal: Calendar) -> bool:
    if d in cal.holidays:
        return False
    if d.weekday() in cal.weekends:
        return False
    return True


RollingConvention = Callable[[date, Calendar], date]


def no_adjustment(d: date, cal: Calendar) -> date:
    return d


def following(d: date, cal: Calendar) -> date:
    """
    Docstring for following
    """
    current: date = d
    while not is_business_day(d=current, cal=cal):
        current += timedelta(days=1)
    return current


def preceding(d: date, cal: Calendar) -> date:
    """
    Docstring for preceding
    """
    current: date = d
    while not is_business_day(d=current, cal=cal):
        current -= timedelta(days=1)
    return current


def modified_following(d: date, cal: Calendar) -> date:
    """
    Docstring for modified_following
    """
    next_biz: date = following(d=d, cal=cal)
    if next_biz.month != d.month:
        return preceding(d=d, cal=cal)
    return next_biz


DayCountConvention = Callable[[date, date], float]


def act_365(start: date, end: date) -> float:
    """
    Docstring for act_365)
    """
    return (end - start).days / 365


def act_360(start: date, end: date) -> float:
    """
    Docstring for act_360)
    """
    return (end - start).days / 360


def thirty_360(start: date, end: date) -> float:
    """
    Docstring for thirty_360)
    """
    d1, m1, y1 = start.day, start.month, start.year
    d2, m2, y2 = end.day, end.month, end.year

    if d1 == 31:
        d1 = 30
    if d2 == 31 and d1 == 30:
        d2 = 30

    days: int = (y2 - y1) * 360 + (m2 - m1) * 30 + (d2 - d1)
    return days / 360


def generate_schedule(
    start: date,
    end: date,
    freq_month: int,
    cal: Calendar,
    convention: RollingConvention,
) -> list[date]:
    """
    Docstring for generate_schedule
    """
    raw_dates: Generator[date, None, None] = (
        start + relativedelta(months=i * freq_month) for i in count()
    )
    future_dates: takewhile[date] = takewhile(lambda d: d <= end, raw_dates)
    adjusted_dates: list[date] = [convention(d, cal) for d in future_dates]
    if adjusted_dates[-1] != convention(end, cal):
        adjusted_dates.append(convention(end, cal))
    return sorted(list(set(adjusted_dates)))
