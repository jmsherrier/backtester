"""Cross-sectional signals: rank the assets against each other.

A time-series signal asks of each asset in isolation, "is *this* trending?" A
cross-sectional signal asks a relative question across the panel at each date —
"which assets are the winners and which the losers *right now*?" — and builds a
market-neutral book from the answer: long the top names, short the bottom ones,
in equal and opposite size. The bet is on the *spread* between winners and
losers, not on the market's direction.

The output is a weight matrix shaped like the input return matrix (one row per
date, one column per asset), ready to feed a multi-asset engine. Every row is
dollar-neutral (weights sum to zero) with unit gross exposure (absolute weights
sum to one), so the leverage is fixed and only the *composition* moves.

The no-lookahead invariant carries over from the time-series case and for the
same reason: each date's ranking uses only that date's trailing window, so data
after t cannot move the weights at t. It is enforced by a truncation-invariance
test, exactly as for ``time_series_momentum``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_panel(returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame):
        raise TypeError(
            f"returns must be a pandas DataFrame, got {type(returns).__name__}"
        )
    if returns.empty:
        raise ValueError("returns is empty")
    if returns.shape[1] < 2:
        raise ValueError(
            f"cross-sectional ranking needs at least 2 assets, got {returns.shape[1]}; "
            "a one-asset panel has no cross-section to rank"
        )
    if returns.isna().to_numpy().any():
        raise ValueError(
            f"returns contains {int(returns.isna().to_numpy().sum())} NaN value(s); "
            "align the panel first (see backtester.data.panel.align_returns)"
        )
    return returns.astype(np.float64)


def cross_sectional_momentum(
    returns: pd.DataFrame, lookback: int = 63, quantile: float = 0.2
) -> pd.DataFrame:
    """Winners-minus-losers weights from trailing cross-sectional momentum.

    At each date the assets are ranked by their trailing ``lookback``-period
    compounded return. The top ``quantile`` fraction are held long in equal
    weight and the bottom ``quantile`` short in equal and opposite weight, so
    every active row is dollar-neutral with unit gross exposure.

    Parameters
    ----------
    returns : pd.DataFrame
        Aligned return matrix, one column per asset, no NaN (see
        :func:`backtester.data.panel.align_returns`).
    lookback : int
        Trailing window length in periods used to rank the assets.
    quantile : float
        Fraction of the universe taken long (and, separately, short), in
        ``(0, 0.5]``. The per-side count is ``max(1, floor(n_assets *
        quantile))``, capped so the long and short legs never overlap — with
        an odd universe the median asset is simply left flat.

    Returns
    -------
    pd.DataFrame
        Target weights aligned to ``returns``. The first ``lookback - 1`` rows
        lack a full window and are flat (all 0.0) — never NaN, so the result
        feeds straight into the engine. Rows are dollar-neutral (sum to 0) and
        unit-gross (absolute values sum to 1) once past warm-up.
    """
    returns = _validate_panel(returns)
    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}")
    if not 0.0 < quantile <= 0.5:
        raise ValueError(f"quantile must be in (0, 0.5], got {quantile}")

    n_assets = returns.shape[1]
    per_side = max(1, int(n_assets * quantile))
    per_side = min(per_side, n_assets // 2)  # keep the legs disjoint

    # Trailing compounded return per asset: sum of log growth shares the sign
    # and ordering of the compounded simple return and rolls in O(n).
    trailing = np.log1p(returns).rolling(lookback).sum()

    # method="first" breaks ties by column order, guaranteeing exactly
    # per_side names on each leg (so gross exposure is exactly 1, net exactly
    # 0). On warm-up rows trailing is all-NaN, the ranks are NaN, both masks
    # are False, and the row is flat — no special-casing needed.
    ranks = trailing.rank(axis=1, method="first")
    leg = 0.5 / per_side
    long_leg = (ranks > n_assets - per_side) * leg
    short_leg = (ranks <= per_side) * leg
    weights = long_leg - short_leg
    return weights.astype(np.float64)
