# PyKwant 🐍📉

> 🚧 **DISCLAIMER: Active Development**
>
> This project is currently in an **Alpha** stage and is under active development. APIs are subject to breaking changes without notice. **PyKwant is NOT ready for production use.**

**PyKwant** is a modern, pure-Python library for Quantitative Finance, built with a strict **Functional Programming** philosophy.

Unlike traditional object-oriented financial libraries where instruments "price themselves" via internal state, PyKwant separates **Data** (Immutable DTOs) from **Behavior** (Pure Functions). This approach leads to code that is easier to test, parallelize, and reason about.

## 🚀 Key Features

* **Functional Paradigm**: Heavily utilizes closures, higher-order functions, and composition.
* **Immutable Data**: All financial instruments and calendars are `frozen dataclass` structures.
* **Type Safety**: Fully typed codebase (`mypy` strict) targeting **Python 3.14+**.
* **Composable**: Designed to work seamlessly with functional libraries like `toolz`.
* **Zero "Magic"**: No global state, no hidden side effects.

## 📦 Installation

PyKwant requires **Python 3.14** or higher.

```bash
# Clone the repository
git clone https://github.com/makoponi/pykwant.git
cd pykwant

# Install dependencies (using uv, pip, or hatch)
pip install .
```

## ⚡ Quick Start

Here is a complete example showing how to model a bond, build a yield curve, and price the instrument.

```python
from datetime import date
from pykwant import instruments, rates, dates

# 1. Define the Market Environment (Yield Curve)
# In PyKwant, a Curve is just a function: Date -> DiscountFactor
ref_date = date(2025, 1, 1)

def simple_curve(d: date) -> float:
    # A simple flat 5% continuous compounding curve
    t = dates.act_365(ref_date, d)
    return rates.compound_factor(rate=-0.05, t=t, frequency=0)

# 2. Define the Instrument (Data)
# Instruments are immutable data structures
bond = instruments.FixedRateBond(
    face_value=instruments.Money(100.0),
    coupon_rate=0.05,
    start_date=date(2025, 1, 1),
    maturity_date=date(2028, 1, 1),
    frequency_months=12,
    day_count=dates.thirty_360,
    calendar=dates.Calendar(holidays=frozenset())
)

# 3. Price the Instrument (Behavior)
# Logic is applied to data externally
valuation_date = date(2025, 1, 1)
price = instruments.price_instrument(bond, simple_curve, valuation_date)

print(f"Bond Price: {price:.4f}")
```

## 🧩 Modules Overview

`pykwant.dates`

Handles time, the most complex variable in finance.

* **Calendars**: Immutable sets of holidays.

* **Rolling Conventions**: `following`, `modified_following`, `preceding`.

* **Day Counts**: `act_365`, `act_360`, `thirty_360`.

`pykwant.rates`

Core interest rate logic.

* **Yield Curves**: Factory functions to create curves from discount factors.

* **Rate Helpers**: `zero_rates`, `forward_rate`, `compound_factor`.

* **Bootstrapping**: (Coming soon) functional bootstrapping logic.

`pykwant.instruments`

Financial product definitions and pricing pipelines.

* **Data Models**: `FixedRateBond`, `CashFlow`.

* **Generators**: `generate_cash_flows` creates deterministic schedules.

* **Pricing**: `price_instrument`, `clean_price`, `accrued_interest`.

`pykwant.numerics`

The mathematical engine.

* **Interpolation**: `linear_interpolation`, `log_linear_interpolation`.

* **Solvers**: `newton_solve` for root finding (e.g., implied volatility, YTM).

* **Differentiation**: `numerical_derivative`.

`pykwant.risk`

Market risk and sensitivity analysis.

* **Metrics**: `calculate_risk_metrics` for Duration, Convexity, and DV01.

* **Methodology**: Uses numerical differentiation on pricing functions.

* **Scenarios**: Supports curve shifting for stress testing.

`pykwant.portfolio`

Portfolio management and aggregation.

* **Structures**: `Position` (Instrument + Quantity).

* **Valuation**: `portfolio_npv` for summing position values.

* **Risk Aggregation**: `portfolio_risk` for total DV01 and Portfolio Duration.

* **Reporting**: `exposure_by_maturity_year`.

## 🛠️ Development

We use modern tooling to ensure code quality.

```bash
# Run tests
pytest

# Run type checking
mypy .

# Format code
ruff format .
```

### Philosophy: Why Functional?

In quantitative finance, state management is often the source of bugs. By treating a Yield Curve as a **Function** rather than an **Object**, we gain flexibility:

* A curve can be a simple interpolation, a complex optimization result, or a mock lambda for testing.

* Pricing becomes a transformation pipeline: `Instrument -> CashFlows -> DiscountedFlows -> Sum`.

* This pipeline is easily inspectable and modifiable using tools like `toolz.pipe`.

## 📄 License

MIT License