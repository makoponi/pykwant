from datetime import date

import pytest

from pykwant.dates import (
    Calendar,
    act_360,
    act_365,
    following,
    generate_schedule,
    is_business_day,
    modified_following,
    preceding,
    thirty_360,
)


@pytest.fixture
def cal() -> Calendar:
    holidays = frozenset([date(2025, 1, 1), date(2025, 12, 25)])
    return Calendar(holidays=holidays)


def test_is_business_day(cal: Calendar) -> None:
    assert is_business_day(date(2025, 1, 2), cal) is True
    assert is_business_day(date(2025, 1, 1), cal) is False
    assert is_business_day(date(2025, 1, 4), cal) is False
    assert is_business_day(date(2025, 12, 25), cal) is False


def test_following(cal: Calendar) -> None:
    d = date(2025, 1, 3)
    assert following(d, cal) == d

    d = date(2025, 1, 4)
    assert following(d, cal) == date(2025, 1, 6)

    d = date(2025, 12, 25)
    assert following(d, cal) == date(2025, 12, 26)


def test_preceding(cal: Calendar) -> None:
    d = date(2025, 1, 3)
    assert preceding(d, cal) == d

    d = date(2025, 1, 5)
    assert preceding(d, cal) == date(2025, 1, 3)

    d = date(2025, 12, 25)
    assert preceding(d, cal) == date(2025, 12, 24)


def test_modified_following(cal: Calendar) -> None:
    d = date(2025, 1, 4)
    assert modified_following(d, cal) == date(2025, 1, 6)

    d = date(2025, 11, 29)
    assert modified_following(d, cal) == date(2025, 11, 28)


def test_day_counts() -> None:
    start = date(2025, 1, 1)
    end = date(2025, 2, 1)

    assert act_365(start, end) == 31 / 365.0
    assert act_360(start, end) == 31 / 360.0

    assert thirty_360(date(2025, 1, 31), date(2025, 2, 28)) == 28 / 360.0


def test_generate_schedule(cal: Calendar) -> None:
    start = date(2025, 1, 1)
    end = date(2025, 4, 1)

    schedule = generate_schedule(start, end, 1, cal, following)

    expected = [
        date(2025, 1, 2),
        date(2025, 2, 3),
        date(2025, 3, 3),
        date(2025, 4, 1),
    ]
    assert schedule == expected
