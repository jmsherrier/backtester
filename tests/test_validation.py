"""Unit tests for backtester.validation.

The load-bearing properties: splits are chronological (every train date
strictly precedes every test date), lossless (the pieces concatenate back
to exactly the original series), and never degenerate (an empty train or
test side raises instead of quietly returning an unsplit series).
"""

import numpy as np
import pandas as pd
import pytest

from backtester.validation import TrainTestSplit, split_by_date, split_by_fraction


def dated_returns(n: int = 100, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2021-01-01", periods=n)
    return pd.Series(rng.normal(0.0005, 0.01, size=n), index=index)


class TestValidation:
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            split_by_fraction(pd.Series(dtype=np.float64))

    def test_not_a_series_raises(self):
        with pytest.raises(TypeError, match="Series"):
            split_by_fraction([1.0, 2.0, 3.0])

    def test_unsorted_index_raises(self):
        series = dated_returns(10).iloc[::-1]
        with pytest.raises(ValueError, match="sorted"):
            split_by_fraction(series)

    def test_duplicate_index_raises(self):
        series = pd.Series([0.01, 0.02], index=pd.to_datetime(["2021-01-04"] * 2))
        with pytest.raises(ValueError, match="duplicate"):
            split_by_fraction(series)


class TestSplitByFraction:
    def test_fraction_bounds(self):
        for bad in (0.0, 1.0, -0.5, 1.5):
            with pytest.raises(ValueError, match="train_fraction"):
                split_by_fraction(dated_returns(), train_fraction=bad)

    def test_sizes(self):
        split = split_by_fraction(dated_returns(100), train_fraction=0.7)
        assert split.n_train == 70
        assert split.n_test == 30

    def test_fraction_too_small_for_nonempty_train_raises(self):
        # floor(3 * 0.1) == 0 -> nothing to fit on
        with pytest.raises(ValueError, match="train"):
            split_by_fraction(dated_returns(3), train_fraction=0.1)


class TestSplitByDate:
    def test_boundary_date_lands_in_train(self):
        series = dated_returns(100)
        boundary = series.index[59]
        split = split_by_date(series, boundary)
        assert split.train.index[-1] == boundary
        assert split.test.index[0] > boundary

    def test_accepts_date_string(self):
        split = split_by_date(dated_returns(100), "2021-03-01")
        assert (split.train.index <= "2021-03-01").all()
        assert (split.test.index > "2021-03-01").all()

    def test_date_between_observations_is_fine(self):
        # A weekend split date belongs to no row; it still partitions cleanly.
        series = dated_returns(100)
        split = split_by_date(series, "2021-03-06")  # a Saturday
        assert split.n_train + split.n_test == len(series)

    def test_date_before_all_data_raises(self):
        with pytest.raises(ValueError, match="train"):
            split_by_date(dated_returns(), "1990-01-01")

    def test_date_after_all_data_raises(self):
        with pytest.raises(ValueError, match="test"):
            split_by_date(dated_returns(), "2030-01-01")


class TestPartitionInvariants:
    """Chronological, lossless, non-overlapping — for both splitters."""

    @pytest.fixture(params=["fraction", "date"])
    def split(self, request) -> TrainTestSplit:
        series = dated_returns(250)
        if request.param == "fraction":
            return split_by_fraction(series, train_fraction=0.6)
        return split_by_date(series, series.index[149])

    def test_train_strictly_precedes_test(self, split):
        assert split.train.index.max() < split.test.index.min()

    def test_pieces_reconstruct_the_original(self, split):
        original = dated_returns(250)
        rebuilt = pd.concat([split.train, split.test])
        pd.testing.assert_series_equal(rebuilt, original)

    def test_no_shared_dates(self, split):
        assert split.train.index.intersection(split.test.index).empty
