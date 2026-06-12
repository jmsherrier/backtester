"""Time-series momentum signals.

The classic trend-following signal: go long when the trailing window's
compounded return is positive, short when negative. Signal values live in
{-1.0, 0.0, +1.0}; the engine applies the execution lag, so the signal for
date t may (and does) use the return observed *on* date t.

The no-lookahead invariant — the signal at t is unchanged by any data after
t — holds by construction here (rolling windows look backward only) and is
enforced by a truncation-invariance test in tests/test_signals.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_returns(returns: pd.Series, name: str = "returns") -> pd.Series:
    if not isinstance(returns, pd.Series):
        raise TypeError(f"{name} must be a pandas Series, got {type(returns).__name__}")
    if returns.empty:
        raise ValueError(f"{name} is empty")
    if returns.isna().any():
        raise ValueError(f"{name} contains {int(returns.isna().sum())} NaN value(s)")
    return returns.astype(np.float64)


def time_series_momentum(returns: pd.Series, lookback: int = 63) -> pd.Series:
    """Sign of the trailing ``lookback``-period compounded return.

    Parameters
    ----------
    returns : pd.Series
        Periodic simple returns of the asset, through date t inclusive.
    lookback : int
        Window length in periods (63 ~ one quarter of daily data).

    Returns
    -------
    pd.Series
        Target weight in {-1.0, 0.0, +1.0}, aligned to ``returns``. The first
        ``lookback - 1`` periods lack a full window and are flat (0.0) — never
        NaN, so the result feeds straight into ``run_backtest``. A trailing
        return of exactly zero is also flat rather than an arbitrary side.
    """
    returns = _validate_returns(returns)
    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}")

    # Sum of log growth has the same sign as the compounded simple return
    # and rolls in O(n). (1 + r) > 0 is guaranteed for any survivable return;
    # a -100% period would already have ended the backtest upstream.
    log_growth = np.log1p(returns)
    trailing = log_growth.rolling(lookback).sum()

    signal = np.sign(trailing)
    signal.iloc[: lookback - 1] = 0.0  # warm-up: no full window yet, stay flat
    return signal
