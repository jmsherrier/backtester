"""Core performance and risk metrics.

All return-based metrics take a :class:`pandas.Series` of *periodic net returns*
(simple returns, already net of costs) indexed by time. Metrics raise on empty
or NaN input rather than silently dropping values — a NaN in a net-return
series almost always means an upstream accounting bug, and hiding it would
defeat the purpose of this project.

Annualization assumes returns are sampled at a fixed frequency given by
``periods_per_year`` (252 for daily equity data).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def _validate_returns(returns: pd.Series) -> pd.Series:
    """Validate a periodic return series, returning it as float64."""
    if not isinstance(returns, pd.Series):
        raise TypeError(f"returns must be a pandas Series, got {type(returns).__name__}")
    if returns.empty:
        raise ValueError("returns is empty")
    if returns.isna().any():
        raise ValueError(
            f"returns contains {int(returns.isna().sum())} NaN value(s); "
            "clean or investigate upstream before computing metrics"
        )
    return returns.astype(np.float64)


def annualized_return(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Compound annual growth rate implied by a periodic return series.

    Computed geometrically: total compounded growth raised to the power of
    (periods_per_year / n_periods), minus one.
    """
    returns = _validate_returns(returns)
    total_growth = float((1.0 + returns).prod())
    if total_growth <= 0.0:
        # Account value hit (or went below) zero; CAGR is undefined.
        return float("nan")
    return total_growth ** (periods_per_year / len(returns)) - 1.0


def annualized_volatility(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Sample standard deviation of periodic returns, scaled by sqrt(time)."""
    returns = _validate_returns(returns)
    if len(returns) < 2:
        return float("nan")
    return float(returns.std(ddof=1)) * np.sqrt(periods_per_year)


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized Sharpe ratio of a periodic net-return series.

    Parameters
    ----------
    returns : pd.Series
        Periodic simple returns, net of costs.
    risk_free_rate : float
        Annual risk-free rate (e.g. ``0.05`` for 5%), converted internally to
        a per-period rate before computing excess returns.
    periods_per_year : int
        Sampling frequency used for annualization.

    Returns
    -------
    float
        Annualized Sharpe ratio, or NaN when volatility is zero (a constant
        return series carries no risk information).
    """
    returns = _validate_returns(returns)
    if len(returns) < 2:
        return float("nan")
    rf_per_period = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = returns - rf_per_period
    vol = float(excess.std(ddof=1))
    if vol == 0.0:
        return float("nan")
    return float(excess.mean()) / vol * np.sqrt(periods_per_year)


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough decline of the compounded equity curve.

    Returns a non-positive number: ``-0.25`` means a 25% drawdown. A series
    that never declines from a peak returns ``0.0``.
    """
    returns = _validate_returns(returns)
    equity = (1.0 + returns).cumprod()
    running_peak = equity.cummax()
    drawdowns = equity / running_peak - 1.0
    return float(drawdowns.min())


def hit_rate(returns: pd.Series) -> float:
    """Fraction of periods with a strictly positive return.

    Flat periods (return exactly zero, e.g. no position held) are excluded
    from the denominator so that a mostly-flat strategy is not rewarded for
    sitting out. Returns NaN if every period is flat.
    """
    returns = _validate_returns(returns)
    active = returns[returns != 0.0]
    if active.empty:
        return float("nan")
    return float((active > 0.0).mean())


def turnover(positions: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualized one-way turnover of a position series.

    ``positions`` holds portfolio weights over time (e.g. ``1.0`` fully long,
    ``-1.0`` fully short, ``0.0`` flat). Turnover is the mean absolute change
    in weight per period, scaled to a yearly figure. The position before the
    first observation is taken to be flat, so entering the initial position
    counts as trading.
    """
    if not isinstance(positions, pd.Series):
        raise TypeError(f"positions must be a pandas Series, got {type(positions).__name__}")
    if positions.empty:
        raise ValueError("positions is empty")
    if positions.isna().any():
        raise ValueError("positions contains NaN value(s)")
    weights = positions.astype(np.float64)
    trades = weights.diff()
    trades.iloc[0] = weights.iloc[0]  # entering from flat is a trade
    return float(trades.abs().mean()) * periods_per_year


def summary(
    returns: pd.Series,
    positions: pd.Series | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    """Compute the standard metric set for a backtest's net-return series.

    Returns a flat dict suitable for tabular reporting. ``turnover`` is
    included only when a position series is provided.
    """
    result = {
        "annualized_return": annualized_return(returns, periods_per_year),
        "annualized_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "hit_rate": hit_rate(returns),
        "n_periods": float(len(_validate_returns(returns))),
    }
    if positions is not None:
        result["turnover"] = turnover(positions, periods_per_year)
    return result
