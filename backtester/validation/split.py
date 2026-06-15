"""Chronological train/test separation.

The contract this module exists to enforce: parameters are chosen on the
train window and the test window is touched exactly once, for the final
report. Splits are always chronological — every train date precedes every
test date — because shuffled splits leak future information into the past
and make any time-series result meaningless.

The splitters cannot stop you from peeking at the test set, but they make
the boundary explicit, validated, and reconstructible: the two pieces
concatenate back to exactly the original series, so nothing is silently
dropped at the seam.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TrainTestSplit:
    """An ordered, non-overlapping partition of one series.

    ``train`` ends where ``test`` begins; concatenating them reproduces the
    original series exactly. Fit parameters on ``train``, evaluate on
    ``test``, and report both numbers so the degradation is visible.
    """

    train: pd.Series
    test: pd.Series

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_test(self) -> int:
        return len(self.test)


def _validate_series(series: pd.Series) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError(f"series must be a pandas Series, got {type(series).__name__}")
    if series.empty:
        raise ValueError("series is empty")
    if not series.index.is_monotonic_increasing:
        raise ValueError(
            "series index must be sorted ascending; a chronological split of "
            "unordered data would not separate past from future"
        )
    if series.index.has_duplicates:
        dupes = series.index[series.index.duplicated()].unique()
        raise ValueError(f"series index has duplicate labels: {list(dupes[:5])}")
    return series


def _build_split(series: pd.Series, n_train: int) -> TrainTestSplit:
    if n_train < 1:
        raise ValueError("split leaves the train set empty")
    if n_train >= len(series):
        raise ValueError("split leaves the test set empty")
    return TrainTestSplit(train=series.iloc[:n_train], test=series.iloc[n_train:])


def split_by_fraction(series: pd.Series, train_fraction: float = 0.7) -> TrainTestSplit:
    """Split chronologically, putting the first ``train_fraction`` in train.

    Parameters
    ----------
    series : pd.Series
        Any time-indexed series (typically returns), sorted ascending with
        unique index labels.
    train_fraction : float
        Fraction of observations assigned to the train window, strictly
        between 0 and 1. The count is floored, and both sides must end up
        non-empty — a "split" with nothing to fit on or nothing to test on
        is an error, not a degenerate success.
    """
    series = _validate_series(series)
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(
            f"train_fraction must be strictly between 0 and 1, got {train_fraction}"
        )
    return _build_split(series, int(len(series) * train_fraction))


def split_by_date(series: pd.Series, split_date) -> TrainTestSplit:
    """Split chronologically at an explicit date.

    ``train`` is every observation dated on or *before* ``split_date``;
    ``test`` is everything after. Pinning the boundary to a calendar date
    (rather than a fraction) is the right choice when the split must align
    with something external — a regime change, a publication date, or the
    moment a strategy was actually conceived.

    Raises if either side comes out empty: a split date outside the data's
    range is almost certainly a mistake, and silently returning the whole
    series as "train" would let in-sample results masquerade as out-of-sample.
    """
    series = _validate_series(series)
    split_date = pd.Timestamp(split_date)
    return _build_split(series, int((series.index <= split_date).sum()))
