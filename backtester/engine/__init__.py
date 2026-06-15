"""The core backtest loop.

Takes a return matrix, a signal, and a cost model, and produces a time series of
positions, trades, gross P&L, costs, and net P&L. The loop is deliberately simple
and auditable: positions are formed from lagged signals so that returns earned on
date t come from positions decided using only data through t-1.
"""

from backtester.engine.backtest import BacktestResult, run_backtest
from backtester.engine.portfolio import (
    PortfolioBacktestResult,
    run_portfolio_backtest,
)

__all__ = [
    "BacktestResult",
    "PortfolioBacktestResult",
    "run_backtest",
    "run_portfolio_backtest",
]
