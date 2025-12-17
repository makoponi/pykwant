from dataclasses import dataclass
from datetime import date
from typing import Any

from toolz import groupby, valmap

from pykwant import instruments, rates, risk


@dataclass(frozen=True)
class Position:
    instrument: Any  # Union[Bond, Swap, etc.]
    quantity: float

    @property
    def direction(self) -> int:
        return 1 if self.quantity > 0 else -1


@dataclass(frozen=True)
class PositionRisk:
    market_value: float
    dv01: float
    contribution: float


Portfolio = list[Position]


def evaluate_position(
    pos: Position,
    curve: rates.YieldCurveFn,
    val_date: date,
) -> float:
    """
    Docstring for evaluate_position
    """
    unit_price = instruments.price_instrument(pos.instrument, curve, val_date)
    return unit_price * pos.quantity


def portfolio_npv(
    portfolio: Portfolio,
    curve: rates.YieldCurveFn,
    val_date: date,
) -> float:
    """
    Docstring for portfolio_npv
    """
    return sum(evaluate_position(p, curve, val_date) for p in portfolio)


def portfolio_risk(
    portfolio: Portfolio,
    curve: rates.YieldCurveFn,
    val_date: date,
) -> dict[str, float]:
    """
    Docstring for portfolio_risk
    """

    def _analyze_pos(pos: Position) -> dict[str, float]:
        metrics = risk.calculate_risk_metrics(
            pos.instrument, curve, val_date, instruments.price_instrument
        )
        mv = metrics["price"] * pos.quantity
        pos_dv01 = metrics["dv01"] * pos.quantity

        return {"market_value": mv, "dv01": pos_dv01, "duration": metrics["duration"]}

    results = [_analyze_pos(p) for p in portfolio]

    total_mv = sum(r["market_value"] for r in results)
    total_dv01 = sum(r["dv01"] for r in results)

    port_duration = (total_dv01 / (total_mv * 0.0001)) if total_mv != 0 else 0

    return {
        "total_market_value": total_mv,
        "total_dv01": total_dv01,
        "portfolio_duration": port_duration,
        "positions_count": len(portfolio),
    }


def exposure_by_maturity_year(
    portfolio: Portfolio,
    curve: rates.YieldCurveFn,
    val_date: date,
) -> dict[int, float]:
    """
    Docstring for exposure_by_maturity_year
    """
    grouped = groupby(lambda p: p.instrument.maturity_date.year, portfolio)
    result: dict[int, float] = valmap(
        lambda group: sum(evaluate_position(p, curve, val_date) for p in group), grouped
    )
    return result
