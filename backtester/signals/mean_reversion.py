"""Time-series mean-reversion signals.

The mirror image of momentum: instead of betting that a recent move continues,
bet that it reverts. The classic Bollinger-style rule fades deviations from a
trailing mean — when the price has stretched far above its recent average, short
it; far below, go long; and when it sits inside a normal band, stay flat. The
"far" is measured in trailing standard deviations (a z-score), so the threshold
adapts to each window's volatility rather than being a fixed return.

Signal values live in {-1.0, 0.0, +1.0}, like ``time_series_momentum``, and on
the same data the two are natural opposites: a steady trend that momentum rides
long is exactly what mean-reversion leans against. Working from returns, the
signal reconstructs a *relative* log-price level (a cumulative sum); the level's
arbitrary starting point washes out because the z-score subtracts the trailing
mean, so no initial price is needed.

The no-lookahead invariant holds by construction — the cumulative level at t and
its trailing window depend only on returns through t — and is enforced by a
truncation-invariance test in tests/test_mean_reversion.py.
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


def mean_reversion(
    returns: pd.Series, lookback: int = 21, entry_z: float = 1.0
) -> pd.Series:
    """Fade trailing deviations from a moving average, gated by a z-score.

    At each date the relative log-price level is compared to its trailing
    ``lookback``-period mean and standard deviation. When the level is at least
    ``entry_z`` standard deviations *above* the mean the signal is short
    (``-1.0``, betting on reversion down); at least ``entry_z`` *below*, long
    (``+1.0``); otherwise flat (``0.0``).

    Parameters
    ----------
    returns : pd.Series
        Periodic simple returns of the asset, through date t inclusive.
    lookback : int
        Window length in periods for the trailing mean and standard deviation
        (21 ~ one trading month of daily data). Must be >= 2.
    entry_z : float
        How many trailing standard deviations the level must reach before a
        position is taken. Larger values trade only on more extreme stretches
        and sit flat more often. Must be >= 0.

    Returns
    -------
    pd.Series
        Target weight in {-1.0, 0.0, +1.0}, aligned to ``returns``. The first
        ``lookback - 1`` periods lack a full window and are flat (0.0) — never
        NaN, so the result feeds straight into ``run_backtest``. A window with
        zero dispersion (a perfectly flat stretch) has no meaningful z-score
        and is also held flat rather than dividing by zero.
    """
    returns = _validate_returns(returns)
    if lookback < 2:
        raise ValueError(f"lookback must be >= 2, got {lookback}")
    if entry_z < 0:
        raise ValueError(f"entry_z must be >= 0, got {entry_z}")

    # Relative log-price level; the cumsum's arbitrary origin cancels in the
    # z-score below, so no starting price is needed. The window looks strictly
    # backward, which is what keeps the signal at t blind to data after t.
    log_price = np.log1p(returns).cumsum()
    window = log_price.rolling(lookback)
    rolling_mean = window.mean()
    rolling_std = window.std(ddof=0)  # population spread of the window itself

    # z is NaN during warm-up (incomplete window) and would be inf/NaN where the
    # window is perfectly flat; the std>0 gate excludes both, leaving those
    # periods flat. -sign(z) is the fade: above the mean -> short, below -> long.
    z = (log_price - rolling_mean) / rolling_std
    extreme = rolling_std.gt(0.0) & z.abs().ge(entry_z)
    signal = pd.Series(0.0, index=returns.index)
    signal[extreme] = -np.sign(z[extreme])
    return signal.astype(np.float64)
