"""
Test suite for pykwant.dates module.
"""

from datetime import date

import pytest

from pykwant import dates

# --- Fixtures ---


@pytest.fixture
def milano_calendar():
    """
    Creates a sample calendar with specific holidays.
    Holidays: Jan 1st (New Year), Jan 6th (Epiphany).
    Weekends: Saturday, Sunday (Default).
    """
    holidays = frozenset([date(2025, 1, 1), date(2025, 1, 6), date(2026, 1, 1), date(2026, 1, 6)])
    return dates.Calendar(holidays=holidays)


# --- Calendar & Business Days Tests ---


def test_calendar_creation():
    cal = dates.Calendar()
    assert isinstance(cal.holidays, frozenset)
    assert cal.weekends == (6, 7)  # Sat, Sun


def test_is_business_day(milano_calendar):
    # Wednesday Jan 1, 2025 -> Holiday
    assert not dates.is_business_day(date(2025, 1, 1), milano_calendar)

    # Saturday Jan 4, 2025 -> Weekend
    assert not dates.is_business_day(date(2025, 1, 4), milano_calendar)

    # Monday Jan 6, 2025 -> Holiday (Epiphany)
    assert not dates.is_business_day(date(2025, 1, 6), milano_calendar)

    # Tuesday Jan 7, 2025 -> Business Day
    assert dates.is_business_day(date(2025, 1, 7), milano_calendar)


# --- Rolling Conventions Tests ---


def test_following(milano_calendar):
    # Jan 4 (Sat) -> Jan 5 (Sun) -> Jan 6 (Mon-Hol) -> Jan 7 (Tue)
    target = date(2025, 1, 4)
    adjusted = dates.following(target, milano_calendar)
    assert adjusted == date(2025, 1, 7)

    # Already business day
    assert dates.following(date(2025, 1, 8), milano_calendar) == date(2025, 1, 8)


def test_preceding(milano_calendar):
    # Jan 4 (Sat) -> Jan 3 (Fri - Biz)
    target = date(2025, 1, 4)
    adjusted = dates.preceding(target, milano_calendar)
    assert adjusted == date(2025, 1, 3)

    # Jan 6 (Mon - Hol) -> Jan 5 (Sun) -> Jan 4 (Sat) -> Jan 3 (Fri)
    # Wait, Preceding from Jan 6:
    # 6 is Hol. 5 is Sun. 4 is Sat. 3 is Fri. Correct.
    target_hol = date(2025, 1, 6)
    assert dates.preceding(target_hol, milano_calendar) == date(2025, 1, 3)


def test_modified_following(milano_calendar):
    # Case 1: Adjustment stays in same month
    # Jan 4 (Sat) -> ... -> Jan 7 (Tue). Both Jan. Behave like Following.
    d1 = date(2025, 1, 4)
    assert dates.modified_following(d1, milano_calendar) == date(2025, 1, 7)

    # Case 2: Adjustment crosses month boundary
    # May 31, 2025 is Saturday.
    # Following would be June 1 (Sun) -> June 2 (Mon).
    # Modified Following must fallback to May 30 (Fri).
    d_end_month = date(2025, 5, 31)
    assert dates.modified_following(d_end_month, milano_calendar) == date(2025, 5, 30)


# --- Day Count Conventions Tests ---


def test_act_365():
    start = date(2025, 1, 1)
    end = date(2025, 2, 1)  # 31 days
    assert dates.act_365(start, end) == 31.0 / 365.0


def test_act_360():
    start = date(2025, 1, 1)
    end = date(2025, 2, 1)  # 31 days
    assert dates.act_360(start, end) == 31.0 / 360.0


def test_thirty_360():
    # Standard case: 2 months
    d1 = date(2025, 1, 1)
    d2 = date(2025, 3, 1)
    # (360*(2025-2025) + 30*(3-1) + (1-1)) / 360 = 60/360
    assert dates.thirty_360(d1, d2) == 60.0 / 360.0

    # 31st Adjustment case
    # Jan 30 to Feb 1
    # D1=30, D2=1. M1=1, M2=2.
    # 30*(1) + (1-30) = 30 - 29 = 1 ?? No.
    # Formula: 360*0 + 30*(2-1) + (1-30) = 30 - 29 = 1 day. Correct.

    # End of month 31st case
    # Jan 31 to Feb 1
    # D1=31 -> becomes 30.
    # 30*(1) + (1-30) = 1 day.
    # So 30/360 treats Jan 30 to Feb 1 SAME as Jan 31 to Feb 1.
    t_30 = dates.thirty_360(date(2025, 1, 30), date(2025, 2, 1))
    t_31 = dates.thirty_360(date(2025, 1, 31), date(2025, 2, 1))
    assert t_30 == t_31


# --- Schedule Generation Tests ---


def test_generate_schedule(milano_calendar):
    # 1 Year, Quarterly, Modified Following
    start = date(2025, 1, 1)
    end = date(2026, 1, 1)

    schedule = dates.generate_schedule(
        start=start, end=end, freq_month=3, cal=milano_calendar, convention=dates.modified_following
    )

    # Expected unadjusted:
    # Apr 1, Jul 1, Oct 1, Jan 1

    # Adjustments:
    # Apr 1 (Tue) -> OK
    # Jul 1 (Tue) -> OK
    # Oct 1 (Wed) -> OK
    # Jan 1 (Thu-Hol) -> Jan 2 (Fri)

    expected = [
        date(2025, 4, 1),
        date(2025, 7, 1),
        date(2025, 10, 1),
        date(2026, 1, 2),  # Shifted due to holiday
    ]

    assert schedule == expected
    assert len(schedule) == 4


def test_schedule_short_stub(milano_calendar):
    # 4 months total, freq 3 months -> 1 full period + 1 stub
    start = date(2025, 1, 1)
    end = date(2025, 5, 1)  # Holiday! May 1st usually劳动节/Labor Day but checks calendar
    # Our milano_calendar doesn't have May 1st as holiday, so it's a Thu.

    schedule = dates.generate_schedule(start, end, 3, milano_calendar)

    # Expected: Apr 1, May 1
    assert schedule == [date(2025, 4, 1), date(2025, 5, 1)]
