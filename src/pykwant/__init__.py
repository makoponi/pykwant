"""
PyKwant Top-level Package

Exposes the core modules for quantitative finance, strictly following a
functional programming paradigm.

Modules:
- dates: Calendar and day count conventions.
- equity: Equity options pricing (Black-Scholes) and models.
- instruments: Financial product definitions (Bonds, Options, etc.).
- math_utils: Lightweight mathematical utilities (PDF, CDF).
- numerics: Numerical methods (interpolation, solvers, differentiation).
- portfolio: Portfolio aggregation and risk analysis.
- rates: Yield curve construction and interest rate helpers.
- risk: Risk metrics (Duration, Convexity, DV01).
"""

from pykwant import (
    dates,
    equity,
    instruments,
    math_utils,
    numerics,
    portfolio,
    rates,
    risk,
)

# Version identifier matches pyproject.toml
__version__ = "0.1.0"

# Explicitly define what is exported by 'from pykwant import *'
__all__ = [
    "dates",
    "equity",
    "instruments",
    "math_utils",
    "numerics",
    "portfolio",
    "rates",
    "risk",
]
