"""Unit tests for backtester.signals.mean_reversion.

Two contracts. First the rule: the signal fades trailing deviations — a stretch
above the moving average shorts, below goes long, inside the band stays flat —
and on a trend it is the exact opposite of momentum. Second, the same
no-lookahead guarantee as every signal here, checked by truncation invariance.
"""

import numpy as np
import pandas as pd
import pytest

from backtester.engine import run_backtest
from backtester.execution import BpsCost
from backtester.signals import mean_reversion, time_series_momentum


def series(*values: float) -> pd.Series:
    return pd.Series(list(values), dtype=np.float64)


def random_returns(n: int = 500, seed: int = 21) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0, 0.01, size=n))


class TestValidation:
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            mean_reversion(pd.Series(dtype=np.float64))

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="NaN"):
            mean_reversion(series(0.01, np.nan, 0.0))

    def test_bad_lookback_raises(self):
        with pytest.raises(ValueError, match="lookback"):
            mean_reversion(series(0.01, 0.02, 0.0), lookback=1)

    def test_negative_entry_z_raises(self):
        with pytest.raises(ValueError, match="entry_z"):
            mean_reversion(series(0.01, 0.02, 0.0), entry_z=-0.5)


class TestReversionDirection:
    def test_uptrend_is_faded_short(self):
        # A steady climb stretches the level above its trailing mean -> short.
        result = mean_reversion(series(0.01, 0.01, 0.01, 0.01), lookback=3, entry_z=1.0)
        assert result.iloc[-1] == -1.0

    def test_downtrend_is_faded_long(self):
        result = mean_reversion(
            series(-0.01, -0.01, -0.01, -0.01), lookback=3, entry_z=1.0
        )
        assert result.iloc[-1] == 1.0

    def test_opposite_of_momentum_on_a_trend(self):
        returns = series(0.01, 0.01, 0.01, 0.01, 0.01)
        rev = mean_reversion(returns, lookback=3, entry_z=1.0)
        mom = time_series_momentum(returns, lookback=3)
        # Where momentum takes a side on this trend, reversion takes the other.
        active = mom != 0.0
        assert (rev[active] == -mom[active]).all()

    def test_flat_history_stays_flat(self):
        # No dispersion in the window -> no z-score -> flat, not a divide-by-zero.
        result = mean_reversion(series(0.0, 0.0, 0.0, 0.0), lookback=3)
        assert (result == 0.0).all()


class TestThreshold:
    def test_below_threshold_is_flat(self):
        # The same trend that trips entry_z=1.0 stays flat at a high threshold.
        returns = series(0.01, 0.01, 0.01, 0.01)
        assert (mean_reversion(returns, lookback=3, entry_z=5.0) == 0.0).all()

    def test_higher_threshold_trades_no_more_often(self):
        returns = random_returns()
        loose = (mean_reversion(returns, lookback=21, entry_z=1.0) != 0).sum()
        strict = (mean_reversion(returns, lookback=21, entry_z=2.0) != 0).sum()
        assert strict <= loose


class TestWarmup:
    def test_warmup_periods_are_flat_not_nan(self):
        result = mean_reversion(random_returns(), lookback=21)
        assert not result.isna().any()
        assert (result.iloc[:20] == 0.0).all()

    def test_values_are_only_minus_one_zero_one(self):
        result = mean_reversion(random_returns(), lookback=21, entry_z=1.0)
        assert set(result.unique()) <= {-1.0, 0.0, 1.0}


class TestNoLookahead:
    """Data after t cannot move the signal at t."""

    def test_truncation_invariance(self):
        returns = random_returns()
        full = mean_reversion(returns, lookback=20, entry_z=1.0)
        for cutoff in (25, 100, 250, 499):
            truncated = mean_reversion(returns.iloc[:cutoff], lookback=20, entry_z=1.0)
            pd.testing.assert_series_equal(full.iloc[:cutoff], truncated)

    def test_changing_the_future_does_not_change_the_past(self):
        returns = random_returns()
        altered = returns.copy()
        altered.iloc[400:] = 0.05  # rewrite the future
        original = mean_reversion(returns, lookback=20)
        rerun = mean_reversion(altered, lookback=20)
        pd.testing.assert_series_equal(original.iloc[:400], rerun.iloc[:400])


class TestEndToEnd:
    def test_feeds_directly_into_engine(self):
        returns = random_returns()
        signal = mean_reversion(returns, lookback=21, entry_z=1.0)
        result = run_backtest(returns, signal, BpsCost())
        report = result.summary()
        assert report["n_periods"] == len(returns)
        assert report["max_drawdown"] <= 0.0
        assert report["turnover"] >= 0.0
