"""Transaction-cost models.

A cost model converts a series of *trades* (period-over-period changes in
portfolio weight) into a series of costs, expressed as a positive return drag
to be subtracted from gross returns. Working in weight/return space keeps the
models independent of account size.

Costs are charged on absolute notional traded: a flip from +1 to -1 trades 2x
notional and pays accordingly. Sign of the trade never matters — there is no
such thing as a rebate for selling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _validate_trades(trades: pd.Series) -> pd.Series:
    """Validate a trade series (weight deltas), returning it as float64."""
    if not isinstance(trades, pd.Series):
        raise TypeError(f"trades must be a pandas Series, got {type(trades).__name__}")
    if trades.empty:
        raise ValueError("trades is empty")
    if trades.isna().any():
        raise ValueError(
            f"trades contains {int(trades.isna().sum())} NaN value(s); "
            "clean or investigate upstream before applying costs"
        )
    return trades.astype(np.float64)


def trades_from_positions(positions: pd.Series) -> pd.Series:
    """Derive the trade series implied by a position (weight) series.

    The position before the first observation is taken to be flat, so the
    initial entry counts as a trade — the same convention used by
    :func:`backtester.metrics.turnover`.
    """
    positions = _validate_trades(positions)  # same shape/NaN requirements
    trades = positions.diff()
    trades.iloc[0] = positions.iloc[0]
    return trades


class CostModel(ABC):
    """Converts trades (weight deltas) into per-period cost as return drag."""

    @abstractmethod
    def cost(self, trades: pd.Series) -> pd.Series:
        """Return the cost of each period's trading.

        Parameters
        ----------
        trades : pd.Series
            Period-over-period changes in portfolio weight.

        Returns
        -------
        pd.Series
            Non-negative costs in return space, aligned to ``trades``.
            Subtract from gross returns to obtain net returns.
        """


class ZeroCost(CostModel):
    """Frictionless trading. Exists as an explicit baseline.

    Comparing a strategy under ``ZeroCost`` against a realistic model makes
    the cost drag visible instead of leaving "before costs" implicit.
    """

    def cost(self, trades: pd.Series) -> pd.Series:
        trades = _validate_trades(trades)
        return pd.Series(0.0, index=trades.index)


@dataclass(frozen=True)
class BpsCost(CostModel):
    """Proportional cost: a flat number of basis points per unit notional traded.

    ``commission_bps`` models broker fees; ``slippage_bps`` models the spread
    crossing / market-impact you pay on top. They are kept as separate
    parameters so studies can report them independently, but both apply to
    every trade.

    The defaults (1 bp commission + 5 bps slippage) are deliberately on the
    pessimistic side for liquid US equities — an honest backtest should have
    to beat conservative costs, not optimistic ones.
    """

    commission_bps: float = 1.0
    slippage_bps: float = 5.0

    def __post_init__(self) -> None:
        if self.commission_bps < 0 or self.slippage_bps < 0:
            raise ValueError(
                f"cost rates must be non-negative, got commission_bps="
                f"{self.commission_bps}, slippage_bps={self.slippage_bps}"
            )

    @property
    def total_rate(self) -> float:
        """Combined cost per unit notional traded, as a fraction."""
        return (self.commission_bps + self.slippage_bps) / 1e4

    def cost(self, trades: pd.Series) -> pd.Series:
        trades = _validate_trades(trades)
        return trades.abs() * self.total_rate
