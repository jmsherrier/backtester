"""The multi-asset backtest loop: a weight matrix -> a long-short book's P&L.

The single-asset engine (:mod:`backtester.engine.backtest`) runs one Series of
weights against one Series of returns. This is the same loop widened to a panel:
a weight *matrix* (one column per asset) against a return matrix of the same
shape. Each asset is lagged, marked, and charged exactly as it would be on its
own, and the per-asset results are summed across the row into one portfolio
return series — so a dollar-neutral book (see
:func:`backtester.signals.cross_sectional.cross_sectional_momentum`) nets its
longs against its shorts period by period.

The two load-bearing decisions carry over unchanged from the single-asset case:

- **The lag that prevents lookahead.** The weight chosen for date t becomes the
  position *held during* t+1, applied per column, so every asset's return on a
  date comes from a decision made with only prior information. The book starts
  flat.
- **Costs are not optional, and they reconcile.** Each asset pays on its own
  notional traded; summed across assets the identity still holds every period:

      net_returns + costs == gross_returns   (to float precision)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.execution.costs import CostModel, ZeroCost
from backtester.metrics.performance import TRADING_DAYS_PER_YEAR, summary


def _validate_matrix(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame, got {type(frame).__name__}")
    if frame.empty:
        raise ValueError(f"{name} is empty")
    if frame.isna().to_numpy().any():
        raise ValueError(
            f"{name} contains {int(frame.isna().to_numpy().sum())} NaN value(s). If "
            "these are signal warm-up periods, fill them with 0.0 (flat) explicitly — "
            "the engine will not guess."
        )
    return frame.astype(np.float64)


@dataclass(frozen=True)
class PortfolioBacktestResult:
    """Time series produced by a multi-asset backtest run.

    ``positions`` and ``trades`` are matrices (one column per asset) — the
    weights actually held during each period (the lagged signal) and the
    period-over-period changes in them. ``gross_returns``, ``costs``, and
    ``net_returns`` are portfolio-level Series, already summed across assets,
    so they reconcile period by period: ``net_returns + costs == gross_returns``.
    """

    positions: pd.DataFrame
    trades: pd.DataFrame
    gross_returns: pd.Series
    costs: pd.Series
    net_returns: pd.Series

    def summary(
        self,
        risk_free_rate: float = 0.0,
        periods_per_year: int = TRADING_DAYS_PER_YEAR,
    ) -> dict[str, float]:
        """Standard metric set on the portfolio's *net* returns.

        ``turnover`` is the book's total one-way turnover — the mean across
        time of the summed absolute weight change over all assets, annualized —
        so rebalancing every leg counts, not just net exposure changes.
        """
        report = summary(
            self.net_returns,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )
        report["turnover"] = (
            float(self.trades.abs().sum(axis=1).mean()) * periods_per_year
        )
        return report


def run_portfolio_backtest(
    returns: pd.DataFrame,
    signal: pd.DataFrame,
    cost_model: CostModel | None = None,
) -> PortfolioBacktestResult:
    """Run a multi-asset backtest of a ``signal`` matrix against ``returns``.

    Parameters
    ----------
    returns : pd.DataFrame
        Periodic simple returns, one column per asset, aligned on a shared
        date index (see :func:`backtester.data.panel.align_returns`).
    signal : pd.DataFrame
        Desired position weights decided at each date using information through
        that date. Must share ``returns``' index and columns exactly. The
        weight for date t is held during t+1 — the engine applies this lag, so
        pass the *unlagged* signal (e.g. straight from a cross-sectional
        ranker).
    cost_model : CostModel, optional
        Defaults to :class:`ZeroCost`. Each asset is charged on its own
        notional traded; pass a realistic model for any number you report.

    Returns
    -------
    PortfolioBacktestResult
    """
    returns = _validate_matrix(returns, "returns")
    signal = _validate_matrix(signal, "signal")
    if not returns.index.equals(signal.index):
        raise ValueError(
            "returns and signal must share an identical index; align them "
            "explicitly before calling run_portfolio_backtest"
        )
    if not returns.columns.equals(signal.columns):
        raise ValueError(
            "returns and signal must share identical columns in the same order; "
            f"got returns={list(returns.columns)} signal={list(signal.columns)}"
        )
    if cost_model is None:
        cost_model = ZeroCost()

    # The lag that prevents lookahead, applied per asset: the weight decided on
    # date t is the position held during t+1. No prior weight exists for the
    # first period, so the book starts flat across every asset.
    positions = signal.shift(1)
    positions.iloc[0] = 0.0

    # Per-asset trades: change in weight, with entry from flat counted on the
    # first period (matching trades_from_positions / metrics.turnover).
    trades = positions.diff()
    trades.iloc[0] = positions.iloc[0]

    # Mark and charge each asset on its own, then sum across the row into one
    # portfolio series. cost() is applied column-wise so any CostModel works.
    gross_returns = (positions * returns).sum(axis=1)
    costs = trades.apply(cost_model.cost).sum(axis=1)
    net_returns = gross_returns - costs

    return PortfolioBacktestResult(
        positions=positions,
        trades=trades,
        gross_returns=gross_returns,
        costs=costs,
        net_returns=net_returns,
    )
