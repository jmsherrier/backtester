"""The core backtest loop: signal -> lagged positions -> net P&L.

The loop is vectorized and deliberately small enough to audit line by line.
The single load-bearing decision is the lag: a signal value at date t becomes
the position *held during* date t+1, so the return earned on any date comes
from a decision made with only prior information. Everything else is
bookkeeping — and the bookkeeping is checked by a reconciliation identity:

    net_returns + costs == gross_returns   (to float precision, every period)

Single-asset for now: ``returns`` and ``signal`` are Series sharing an index.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.execution.costs import CostModel, ZeroCost, trades_from_positions
from backtester.metrics.performance import TRADING_DAYS_PER_YEAR, summary


def _validate_input(series: pd.Series, name: str) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series, got {type(series).__name__}")
    if series.empty:
        raise ValueError(f"{name} is empty")
    if series.isna().any():
        raise ValueError(
            f"{name} contains {int(series.isna().sum())} NaN value(s). If these are "
            "signal warm-up periods, fill them with 0.0 (flat) explicitly — the "
            "engine will not guess."
        )
    return series.astype(np.float64)


@dataclass(frozen=True)
class BacktestResult:
    """Time series produced by a backtest run, all sharing one index.

    ``positions`` is the weight actually held during each period (i.e. the
    lagged signal), not the signal itself — so ``positions * asset returns``
    is exactly ``gross_returns``.
    """

    positions: pd.Series
    trades: pd.Series
    gross_returns: pd.Series
    costs: pd.Series
    net_returns: pd.Series

    def summary(
        self,
        risk_free_rate: float = 0.0,
        periods_per_year: int = TRADING_DAYS_PER_YEAR,
    ) -> dict[str, float]:
        """Standard metric set on *net* returns. Costs are never optional."""
        return summary(
            self.net_returns,
            positions=self.positions,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )


def run_backtest(
    returns: pd.Series,
    signal: pd.Series,
    cost_model: CostModel | None = None,
) -> BacktestResult:
    """Run a single-asset backtest of ``signal`` against ``returns``.

    Parameters
    ----------
    returns : pd.Series
        Periodic simple returns of the traded asset.
    signal : pd.Series
        Desired position weight decided at each date using information
        available through that date (e.g. ``1.0`` long, ``-1.0`` short,
        ``0.0`` flat). Must share ``returns``' index exactly. The signal for
        date t is held during date t+1 — the engine applies this lag itself,
        so pass the *unlagged* signal.
    cost_model : CostModel, optional
        Defaults to :class:`ZeroCost`. Pass a realistic model (e.g.
        ``BpsCost()``) for any number you intend to report.

    Returns
    -------
    BacktestResult
    """
    returns = _validate_input(returns, "returns")
    signal = _validate_input(signal, "signal")
    if not returns.index.equals(signal.index):
        raise ValueError(
            "returns and signal must share an identical index; align them "
            "explicitly before calling run_backtest"
        )
    if cost_model is None:
        cost_model = ZeroCost()

    # The lag that prevents lookahead: the signal decided on date t is the
    # position held during date t+1. There is no prior signal for the first
    # period, so the strategy starts flat.
    positions = signal.shift(1)
    positions.iloc[0] = 0.0

    gross_returns = positions * returns
    trades = trades_from_positions(positions)
    costs = cost_model.cost(trades)
    net_returns = gross_returns - costs

    return BacktestResult(
        positions=positions,
        trades=trades,
        gross_returns=gross_returns,
        costs=costs,
        net_returns=net_returns,
    )
