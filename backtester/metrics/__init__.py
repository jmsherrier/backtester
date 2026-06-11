"""Performance and risk metrics.

Annualized Sharpe, max drawdown, turnover, hit rate, and volatility. All metrics
operate on a net-return series produced by the engine so that costs are always
reflected in the reported numbers.
"""

from backtester.metrics.performance import (
    TRADING_DAYS_PER_YEAR,
    annualized_return,
    annualized_volatility,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
    summary,
    turnover,
)

__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "annualized_return",
    "annualized_volatility",
    "hit_rate",
    "max_drawdown",
    "sharpe_ratio",
    "summary",
    "turnover",
]
