"""
PyKwant Top-level Package

Exposes the core modules for quantitative finance:
- dates: Calendar and day count conventions.
- rates: Yield curve construction and rate helpers.
- instruments: Financial product definitions (Bonds, etc.).
- numerics: Numerical methods (interpolation, solvers).
- risk: Risk metrics (Duration, Convexity, DV01).
- portfolio: Portfolio aggregation and analysis.
"""

from pykwant import dates, instruments, numerics, portfolio, rates, risk

# Version identifier matches pyproject.toml
__version__ = "0.1.0"

# Explicitly define what is exported by 'from pykwant import *'
__all__ = [
    "dates",
    "instruments",
    "numerics",
    "portfolio",
    "rates",
    "risk",
]
