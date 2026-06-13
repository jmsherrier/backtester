"""Train/test separation and out-of-sample evaluation.

Splits are always chronological — every train date precedes every test
date — and the test window is meant to be touched exactly once, for the
final report. The headline number this package exists to produce is the
out-of-sample Sharpe after costs, shown next to the in-sample one.
"""

from backtester.validation.split import (
    TrainTestSplit,
    split_by_date,
    split_by_fraction,
)

__all__ = [
    "TrainTestSplit",
    "split_by_date",
    "split_by_fraction",
]
