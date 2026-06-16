"""Strategy signal generators.

A signal maps a price/return history available up to and including time t into a
desired target position for t+1. The critical invariant: a signal computed for
date t may only use information observable on or before date t. Enforced by tests.
"""

from backtester.signals.cross_sectional import cross_sectional_momentum
from backtester.signals.mean_reversion import mean_reversion
from backtester.signals.momentum import time_series_momentum

__all__ = [
    "cross_sectional_momentum",
    "mean_reversion",
    "time_series_momentum",
]
