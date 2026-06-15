"""Unit tests for backtester.validation.

The load-bearing properties: splits are chronological (every train date
strictly precedes every test date), lossless (the pieces concatenate back
to exactly the original series), and never degenerate (an empty train or
test side raises instead of quietly returning an unsplit series).
"""

import numpy as np
import pandas as pd
import pytest

from backtester.data import generate_gbm_panel, prices_to_returns
from backtester.engine import run_portfolio_backtest
from backtester.execution import BpsCost
from backtester.signals import cross_sectional_momentum, time_series_momentum
from backtester.validation import (
    TrainTestSplit,
    out_of_sample_study,
    split_by_date,
    split_by_fraction,
    walk_forward,
)


def dated_returns(n: int = 100, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2021-01-01", periods=n)
    return pd.Series(rng.normal(0.0005, 0.01, size=n), index=index)


def dated_panel(n_assets: int = 5, n: int = 200, seed: int = 7) -> pd.DataFrame:
    return prices_to_returns(
        generate_gbm_panel(n_assets=n_assets, n_periods=n + 1, seed=seed)
    )


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


def always(weight: float):
    """Constant-weight signal builder, for studies with a known right answer."""
    return lambda returns: pd.Series(weight, index=returns.index)


class TestOutOfSampleStudy:
    def test_empty_candidates_raise(self):
        with pytest.raises(ValueError, match="empty"):
            out_of_sample_study(dated_returns(), {})

    def test_all_flat_candidates_raise(self):
        # A flat strategy has zero volatility -> NaN Sharpe -> nothing to select.
        with pytest.raises(ValueError, match="finite"):
            out_of_sample_study(dated_returns(), {"flat": always(0.0)})

    def test_selects_the_better_candidate_on_train(self):
        # Strong positive drift: long beats short in-sample, decisively.
        rng = np.random.default_rng(11)
        returns = pd.Series(
            rng.normal(0.002, 0.01, size=400),
            index=pd.bdate_range("2021-01-01", periods=400),
        )
        result = out_of_sample_study(
            returns, {"long": always(1.0), "short": always(-1.0)}
        )
        assert result.selected == "long"
        assert result.train_scores["long"] > result.train_scores["short"]
        assert set(result.train_scores) == {"long", "short"}

    def test_windows_have_the_split_sizes(self):
        returns = dated_returns(200)
        result = out_of_sample_study(
            returns, {"long": always(1.0)}, train_fraction=0.7
        )
        assert result.in_sample["n_periods"] == 140
        assert result.out_of_sample["n_periods"] == 60

    def test_signal_history_spans_the_split_boundary(self):
        # A builder whose first 130 outputs are warm-up. The test window is
        # only 120 periods long, so if the study (wrongly) rebuilt the signal
        # from the test slice alone, the whole window would be flat and the
        # out-of-sample Sharpe NaN. Built on the full history — as a real
        # trader's signal would be — the test window is past warm-up.
        def slow_warmup(returns: pd.Series) -> pd.Series:
            signal = pd.Series(1.0, index=returns.index)
            signal.iloc[:130] = 0.0
            return signal

        result = out_of_sample_study(
            dated_returns(400), {"slow": slow_warmup}, train_fraction=0.7
        )
        assert np.isfinite(result.out_of_sample["sharpe_ratio"])

    def test_runs_with_a_real_signal(self):
        # End-to-end with momentum over a lookback grid: structural checks
        # only — on random data the *values* are noise by design.
        returns = dated_returns(500)
        candidates = {
            lookback: (lambda r, lb=lookback: time_series_momentum(r, lb))
            for lookback in (10, 21, 63)
        }
        result = out_of_sample_study(returns, candidates)
        assert result.selected in candidates
        assert result.in_sample["max_drawdown"] <= 0.0
        assert result.out_of_sample["n_periods"] == 150


class TestWalkForward:
    def test_empty_candidates_raise(self):
        with pytest.raises(ValueError, match="empty"):
            walk_forward(dated_returns(), {}, train_size=50, test_size=10)

    def test_nonpositive_sizes_raise(self):
        for train, test in ((0, 10), (10, 0)):
            with pytest.raises(ValueError, match="positive"):
                walk_forward(
                    dated_returns(), {"long": always(1.0)},
                    train_size=train, test_size=test,
                )

    def test_too_few_periods_for_one_fold_raises(self):
        with pytest.raises(ValueError, match="one fold"):
            walk_forward(
                dated_returns(50), {"long": always(1.0)},
                train_size=60, test_size=10,
            )

    def test_all_flat_candidates_raise(self):
        with pytest.raises(ValueError, match="finite"):
            walk_forward(
                dated_returns(100), {"flat": always(0.0)},
                train_size=50, test_size=10,
            )

    def test_oos_track_is_continuous_and_covers_all_post_train_dates(self):
        # Every period after the initial train window appears exactly once,
        # in order, with no gaps and no overlap between folds.
        returns = dated_returns(200)
        result = walk_forward(
            returns, {"long": always(1.0)}, train_size=100, test_size=20
        )
        expected = returns.index[100:]
        pd.testing.assert_index_equal(result.oos_returns.index, expected)
        assert result.oos_returns.index.is_monotonic_increasing
        assert not result.oos_returns.index.has_duplicates

    def test_uneven_division_keeps_a_short_final_fold(self):
        # 100 train + 95 remaining at step 20 -> folds of 20,20,20,20,15.
        returns = dated_returns(195)
        result = walk_forward(
            returns, {"long": always(1.0)}, train_size=100, test_size=20
        )
        assert [f.n_test for f in result.folds] == [20, 20, 20, 20, 15]
        assert sum(f.n_test for f in result.folds) == 95

    def test_anchored_train_window_grows_rolling_stays_fixed(self):
        returns = dated_returns(200)
        candidates = {"long": always(1.0)}
        anchored = walk_forward(
            returns, candidates, train_size=100, test_size=20, anchored=True
        )
        rolling = walk_forward(
            returns, candidates, train_size=100, test_size=20, anchored=False
        )
        assert [f.n_train for f in anchored.folds] == [100, 120, 140, 160, 180]
        assert [f.n_train for f in rolling.folds] == [100, 100, 100, 100, 100]

    def test_no_flat_reset_at_fold_boundaries(self):
        # A constant-long strategy never trades after entry. If each fold were
        # run in isolation it would re-enter (and pay cost) at every boundary,
        # inflating turnover. Continuous handling means one entry, near-zero
        # turnover across the whole track.
        returns = dated_returns(200)
        result = walk_forward(
            returns, {"long": always(1.0)}, train_size=100, test_size=20
        )
        # Constant position held the entire out-of-sample track.
        assert (result.oos_positions == 1.0).all()

    def test_selection_can_change_across_folds(self):
        # Returns flip from strong up-drift to strong down-drift halfway; the
        # rolling refit should switch from long to short as the regime turns.
        rng = np.random.default_rng(3)
        up = rng.normal(0.003, 0.01, size=120)
        down = rng.normal(-0.003, 0.01, size=120)
        returns = pd.Series(
            np.concatenate([up, down]),
            index=pd.bdate_range("2021-01-01", periods=240),
        )
        result = walk_forward(
            returns, {"long": always(1.0), "short": always(-1.0)},
            train_size=60, test_size=30, anchored=False,
        )
        selections = [f.selected for f in result.folds]
        assert "long" in selections and "short" in selections

    def test_runs_with_a_real_signal(self):
        returns = dated_returns(600)
        candidates = {
            lookback: (lambda r, lb=lookback: time_series_momentum(r, lb))
            for lookback in (21, 63)
        }
        result = walk_forward(returns, candidates, train_size=300, test_size=60)
        assert result.summary["max_drawdown"] <= 0.0
        assert result.summary["n_periods"] == len(result.oos_returns)
        assert all(f.selected in candidates for f in result.folds)


class TestSplitPanel:
    """The splitters partition a return matrix by date, keeping every column."""

    def test_fraction_splits_a_dataframe_by_row(self):
        panel = dated_panel(n_assets=5, n=200)
        split = split_by_fraction(panel, train_fraction=0.7)
        assert isinstance(split.train, pd.DataFrame)
        assert split.n_train == 140 and split.n_test == 60
        assert list(split.train.columns) == list(panel.columns)

    def test_date_splits_a_dataframe_by_row(self):
        panel = dated_panel(n_assets=4, n=200)
        boundary = panel.index[119]
        split = split_by_date(panel, boundary)
        assert split.train.index[-1] == boundary
        assert split.test.index[0] > boundary
        assert split.train.shape[1] == split.test.shape[1] == 4

    def test_panel_split_is_chronological_and_lossless(self):
        panel = dated_panel(n_assets=6, n=250)
        split = split_by_fraction(panel, train_fraction=0.6)
        assert split.train.index.max() < split.test.index.min()
        pd.testing.assert_frame_equal(pd.concat([split.train, split.test]), panel)

    def test_unsorted_panel_raises(self):
        with pytest.raises(ValueError, match="sorted"):
            split_by_fraction(dated_panel(n=50).iloc[::-1])


class TestMultiAssetStudyReadiness:
    """Smoke test of the capability the next milestone needs: a chronological
    split of a return matrix, the multi-asset engine evaluated once on the
    held-out block. Proves the pieces compose before that study is built."""

    def test_split_then_portfolio_backtest_on_held_out_block(self):
        returns = dated_panel(n_assets=10, n=600, seed=3)
        split = split_by_fraction(returns, train_fraction=0.7)

        # Signal built on the full history (legitimately past at any test date),
        # evaluated only over the held-out block — the same boundary discipline
        # out_of_sample_study uses for the single-asset case.
        signal = cross_sectional_momentum(returns, lookback=63)
        result = run_portfolio_backtest(
            split.test, signal.loc[split.test.index], BpsCost()
        )
        report = result.summary()
        assert report["n_periods"] == split.n_test
        assert report["max_drawdown"] <= 0.0
        assert np.isfinite(report["sharpe_ratio"])
        # Held-out block is dollar-neutral throughout (longs net against shorts).
        assert np.allclose(result.positions.sum(axis=1).to_numpy(), 0.0)
