"""
Test suite for pykwant.bootstrapping module.
"""

import math
from datetime import date

import pytest

from pykwant import bootstrapping, dates, rates

# --- Fixtures ---


@pytest.fixture
def val_date():
    return date(2025, 1, 1)


@pytest.fixture
def sample_helpers(val_date):
    """
    Set of market data:
    - Deposit 6M: 2.0%
    - Swap 1Y: 2.5%
    - Swap 2Y: 3.0%
    """
    cal = dates.Calendar()

    # 6M Deposit
    dep_mat = date(2025, 7, 1)
    dep = bootstrapping.DepositRate(0.02, dep_mat)

    # 1Y Swap
    swap1_mat = date(2026, 1, 1)
    swap1 = bootstrapping.SwapRate(0.025, swap1_mat, 12, calendar=cal)

    # 2Y Swap
    swap2_mat = date(2027, 1, 1)
    swap2 = bootstrapping.SwapRate(0.030, swap2_mat, 12, calendar=cal)

    return [dep, swap1, swap2]


# --- Tests ---


def test_bootstrap_repricing(val_date, sample_helpers):
    # Build curve
    curve = bootstrapping.bootstrap_curve(val_date, sample_helpers)

    # 1. Check Deposit Repricing
    # Deposit PV should be 1.0
    dep = sample_helpers[0]
    pv_dep = bootstrapping._price_helper(dep, curve, val_date)
    assert math.isclose(pv_dep, 1.0, rel_tol=1e-6)

    # 2. Check Swap 1Y Repricing
    # Swap PV (Fixed Bond) should be 1.0 (Par)
    swap1 = sample_helpers[1]
    pv_swap1 = bootstrapping._price_helper(swap1, curve, val_date)
    assert math.isclose(pv_swap1, 1.0, rel_tol=1e-6)

    # 3. Check Swap 2Y Repricing
    swap2 = sample_helpers[2]
    pv_swap2 = bootstrapping._price_helper(swap2, curve, val_date)
    assert math.isclose(pv_swap2, 1.0, rel_tol=1e-6)


def test_bootstrap_curve_shape(val_date, sample_helpers):
    curve = bootstrapping.bootstrap_curve(val_date, sample_helpers)

    # Check Zero Rates at pillars
    # 6M: Rate should be close to 2% (slightly different due to ACT/360 vs ACT/365)
    r_6m = rates.zero_rates(curve, val_date, sample_helpers[0].maturity_date)
    assert math.isclose(r_6m, 0.02, abs_tol=0.001)

    # 2Y: Should be higher (around 3%)
    r_2y = rates.zero_rates(curve, val_date, sample_helpers[2].maturity_date)
    assert math.isclose(r_2y, 0.03, abs_tol=0.001)

    # Check monotonicity (Normal upward sloping curve)
    assert r_2y > r_6m


def test_bootstrap_empty_error(val_date):
    # If no helpers, should probably just return flat 1.0 or fail?
    # Our implementation returns a curve with just the ref date.
    curve = bootstrapping.bootstrap_curve(val_date, [])
    assert curve(val_date) == 1.0
