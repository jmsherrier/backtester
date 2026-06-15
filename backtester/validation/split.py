"""Chronological train/test separation.

The contract this module exists to enforce: parameters are chosen on the
train window and the test window is touched exactly once, for the final
report. Splits are always chronological — every train date precedes every
test date — because shuffled splits leak future information into the past
and make any time-series result meaningless.

The splitters cannot stop you from peeking at the test set, but they make
the boundary explicit, validated, and reconstructible: the two pieces
concatenate back to exactly the original data, so nothing is silently
dropped at the seam.

Both splitters accept a single asset (``Series``) or an aligned return matrix
(``DataFrame``) — a panel is split by date, the same boundary applied to every
column at once, which is what an out-of-sample or walk-forward study driven by
the multi-asset engine needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

Data = pd.Series | pd.DataFrame


@dataclass(frozen=True)
class TrainTestSplit:
    """An ordered, non-overlapping partition of one series or panel.

    ``train`` ends where ``test`` begins; concatenating them reproduces the
    original data exactly. Fit parameters on ``train``, evaluate on ``test``,
    and report both numbers so the degradation is visible. For a panel both
    sides carry every asset column; only the rows (dates) are partitioned.
    """

    train: Data
    test: Data

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_test(self) -> int:
        return len(self.test)


def _validate(data: Data) -> Data:
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError(
            f"data must be a pandas Series or DataFrame, got {type(data).__name__}"
        )
    if data.empty:
        raise ValueError("data is empty")
    if not data.index.is_monotonic_increasing:
        raise ValueError(
            "data index must be sorted ascending; a chronological split of "
            "unordered data would not separate past from future"
        )
    if data.index.has_duplicates:
        dupes = data.index[data.index.duplicated()].unique()
        raise ValueError(f"data index has duplicate labels: {list(dupes[:5])}")
    return data


def _build_split(data: Data, n_train: int) -> TrainTestSplit:
    if n_train < 1:
        raise ValueError("split leaves the train set empty")
    if n_train >= len(data):
        raise ValueError("split leaves the test set empty")
    return TrainTestSplit(train=data.iloc[:n_train], test=data.iloc[n_train:])


def split_by_fraction(data: Data, train_fraction: float = 0.7) -> TrainTestSplit:
    """Split chronologically, putting the first ``train_fraction`` in train.

    Parameters
    ----------
    data : pd.Series | pd.DataFrame
        Any time-indexed series or panel (typically returns), sorted ascending
        with unique index labels. A panel is split by row (date).
    train_fraction : float
        Fraction of observations assigned to the train window, strictly
        between 0 and 1. The count is floored, and both sides must end up
        non-empty — a "split" with nothing to fit on or nothing to test on
        is an error, not a degenerate success.
    """
    data = _validate(data)
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(
            f"train_fraction must be strictly between 0 and 1, got {train_fraction}"
        )
    return _build_split(data, int(len(data) * train_fraction))


def split_by_date(data: Data, split_date) -> TrainTestSplit:
    """Split chronologically at an explicit date.

    ``train`` is every observation dated on or *before* ``split_date``;
    ``test`` is everything after. Pinning the boundary to a calendar date
    (rather than a fraction) is the right choice when the split must align
    with something external — a regime change, a publication date, or the
    moment a strategy was actually conceived.

    Raises if either side comes out empty: a split date outside the data's
    range is almost certainly a mistake, and silently returning the whole
    dataset as "train" would let in-sample results masquerade as out-of-sample.
    """
    data = _validate(data)
    split_date = pd.Timestamp(split_date)
    return _build_split(data, int((data.index <= split_date).sum()))
