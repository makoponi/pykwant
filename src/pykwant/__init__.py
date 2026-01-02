"""
PyKwant Top-level Package

Exposes the core modules for quantitative finance, strictly following a
functional programming paradigm.

Modules:
- analytics: Performance analysis (Returns, Sharpe, Drawdown).
- dates: Calendar and day count conventions.
- equity: Equity options pricing (Black-Scholes) and models.
- instruments: Financial product definitions (Bonds, Options, etc.).
- market_risk: Market risk metrics (VaR, ES).
- math_utils: Lightweight mathematical utilities (PDF, CDF).
- numerics: Numerical methods (interpolation, solvers, differentiation).
- portfolio: Portfolio aggregation and risk analysis.
- rates: Yield curve construction and interest rate helpers.
- risk: Risk metrics (Duration, Convexity, DV01).
- trees: Binomial trees.
"""

from pykwant import (
    analytics,
    dates,
    equity,
    instruments,
    market_risk,
    math_utils,
    numerics,
    portfolio,
    rates,
    risk,
    trees,
)

# Version identifier matches pyproject.toml
__version__ = "0.1.0"

# Explicitly define what is exported by 'from pykwant import *'
__all__ = [
    "analytics",
    "dates",
    "equity",
    "instruments",
    "market_risk",
    "math_utils",
    "numerics",
    "portfolio",
    "rates",
    "risk",
    "trees",
]
