"""Unit tests for backtester.signals.cross_sectional.

Two load-bearing contracts. First, the book is well-formed: every active row
is dollar-neutral (weights sum to zero) with unit gross exposure, longs are the
trailing winners and shorts the losers. Second, the same no-lookahead guarantee
the time-series signal carries — data after t cannot move the weights at t —
checked here by truncation invariance across the panel.
"""

import numpy as np
import pandas as pd
import pytest

from backtester.data import generate_gbm_panel, prices_to_returns
from backtester.signals import cross_sectional_momentum


def panel(n_assets: int = 6, n_periods: int = 400, seed: int = 1) -> pd.DataFrame:
    return prices_to_returns(
        generate_gbm_panel(n_assets=n_assets, n_periods=n_periods, seed=seed)
    )


class TestValidation:
    def test_not_a_dataframe_raises(self):
        with pytest.raises(TypeError, match="DataFrame"):
            cross_sectional_momentum(pd.Series([0.01, 0.02]))

    def test_single_asset_raises(self):
        one = panel(n_assets=6).iloc[:, [0]]
        with pytest.raises(ValueError, match="at least 2 assets"):
            cross_sectional_momentum(one)

    def test_nan_raises(self):
        returns = panel()
        returns.iloc[3, 1] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            cross_sectional_momentum(returns)

    def test_bad_lookback_raises(self):
        with pytest.raises(ValueError, match="lookback"):
            cross_sectional_momentum(panel(), lookback=0)

    def test_bad_quantile_raises(self):
        for bad in (0.0, 0.6, -0.1, 1.0):
            with pytest.raises(ValueError, match="quantile"):
                cross_sectional_momentum(panel(), quantile=bad)


class TestBookShape:
    def test_same_shape_and_labels_as_input(self):
        returns = panel()
        weights = cross_sectional_momentum(returns, lookback=63)
        assert weights.shape == returns.shape
        assert list(weights.columns) == list(returns.columns)
        pd.testing.assert_index_equal(weights.index, returns.index)

    def test_no_nan_anywhere(self):
        weights = cross_sectional_momentum(panel(), lookback=63)
        assert not weights.isna().to_numpy().any()

    def test_active_rows_are_dollar_neutral(self):
        weights = cross_sectional_momentum(panel(), lookback=63)
        active = weights.iloc[63:]
        row_sums = active.sum(axis=1)
        assert np.allclose(row_sums, 0.0)

    def test_active_rows_have_unit_gross_exposure(self):
        weights = cross_sectional_momentum(panel(), lookback=63)
        active = weights.iloc[63:]
        gross = active.abs().sum(axis=1)
        assert np.allclose(gross, 1.0)

    def test_per_side_count_follows_quantile(self):
        # 10 assets, quantile 0.2 -> 2 long + 2 short on every active row.
        weights = cross_sectional_momentum(panel(n_assets=10), lookback=20, quantile=0.2)
        row = weights.iloc[100]
        assert (row > 0).sum() == 2
        assert (row < 0).sum() == 2

    def test_odd_universe_leaves_the_median_flat(self):
        # 5 assets, quantile 0.5 -> 2 per side, capped to keep legs disjoint,
        # so exactly one asset sits flat each active row.
        weights = cross_sectional_momentum(panel(n_assets=5), lookback=20, quantile=0.5)
        row = weights.iloc[100]
        assert (row > 0).sum() == 2
        assert (row < 0).sum() == 2
        assert (row == 0).sum() == 1


class TestRankingDirection:
    def test_longs_the_winner_shorts_the_loser(self):
        # Three assets with hand-built, clearly ordered trailing returns: A
        # rises fastest, C falls. With one name per side, A is long, C short.
        index = pd.bdate_range("2021-01-04", periods=3)
        returns = pd.DataFrame(
            {"A": [0.05, 0.05, 0.05], "B": [0.0, 0.0, 0.0], "C": [-0.05, -0.05, -0.05]},
            index=index,
        )
        weights = cross_sectional_momentum(returns, lookback=2, quantile=0.3)
        last = weights.iloc[-1]
        assert last["A"] > 0
        assert last["C"] < 0
        assert last["B"] == 0.0


class TestWarmup:
    def test_warmup_rows_are_flat_not_nan(self):
        weights = cross_sectional_momentum(panel(), lookback=63)
        warmup = weights.iloc[:62]
        assert not warmup.isna().to_numpy().any()
        assert (warmup == 0.0).to_numpy().all()

    def test_first_full_window_is_active(self):
        weights = cross_sectional_momentum(panel(n_assets=4), lookback=20)
        assert (weights.iloc[18] == 0.0).all()
        assert weights.iloc[19].abs().sum() == pytest.approx(1.0)


class TestNoLookahead:
    """Data after t cannot move the cross-sectional weights at t."""

    def test_truncation_invariance(self):
        returns = panel(n_assets=8, n_periods=500, seed=4)
        full = cross_sectional_momentum(returns, lookback=20)
        for cutoff in (25, 100, 250, 499):
            truncated = cross_sectional_momentum(returns.iloc[:cutoff], lookback=20)
            pd.testing.assert_frame_equal(full.iloc[:cutoff], truncated)

    def test_changing_the_future_does_not_change_the_past(self):
        returns = panel(n_assets=8, n_periods=500, seed=4)
        altered = returns.copy()
        altered.iloc[400:] = -0.05  # rewrite the future
        original = cross_sectional_momentum(returns, lookback=20)
        rerun = cross_sectional_momentum(altered, lookback=20)
        pd.testing.assert_frame_equal(original.iloc[:400], rerun.iloc[:400])
