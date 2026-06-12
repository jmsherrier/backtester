"""Unit tests for backtester.signals.

The load-bearing test here is truncation invariance: a signal computed for
date t must be identical whether or not any data after t exists. That is the
module's no-lookahead contract, checked directly rather than assumed.
"""

import numpy as np
import pandas as pd
import pytest

from backtester.engine import run_backtest
from backtester.execution import BpsCost
from backtester.signals import time_series_momentum


def series(*values: float) -> pd.Series:
    return pd.Series(list(values), dtype=np.float64)


def random_returns(n: int = 500, seed: int = 21) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0005, 0.01, size=n))


class TestValidation:
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            time_series_momentum(pd.Series(dtype=np.float64))

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="NaN"):
            time_series_momentum(series(0.01, np.nan))

    def test_bad_lookback_raises(self):
        with pytest.raises(ValueError, match="lookback"):
            time_series_momentum(series(0.01, 0.02), lookback=0)


class TestMomentumDirection:
    def test_uptrend_goes_long(self):
        # 3-period trailing return after three +1% days is positive -> long
        result = time_series_momentum(series(0.01, 0.01, 0.01, 0.01), lookback=3)
        assert result.iloc[-1] == 1.0

    def test_downtrend_goes_short(self):
        result = time_series_momentum(series(-0.01, -0.01, -0.01, -0.01), lookback=3)
        assert result.iloc[-1] == -1.0

    def test_sign_uses_compounded_not_summed_return(self):
        # +10% then -10% compounds to 0.99 - 1 < 0: a loss, so short — even
        # though the arithmetic sum of returns is exactly zero.
        result = time_series_momentum(series(0.10, -0.10), lookback=2)
        assert result.iloc[-1] == -1.0

    def test_flat_history_stays_flat(self):
        result = time_series_momentum(series(0.0, 0.0, 0.0), lookback=2)
        assert (result == 0.0).all()


class TestWarmup:
    def test_warmup_periods_are_flat_not_nan(self):
        result = time_series_momentum(random_returns(), lookback=63)
        assert not result.isna().any()
        assert (result.iloc[:62] == 0.0).all()

    def test_first_full_window_produces_a_view(self):
        # with lookback=2, period 1 (0-indexed) is the first with a full window
        result = time_series_momentum(series(0.01, 0.01, 0.01), lookback=2)
        assert result.iloc[0] == 0.0
        assert result.iloc[1] == 1.0

    def test_lookback_one_has_no_warmup(self):
        result = time_series_momentum(series(0.01, -0.01), lookback=1)
        assert list(result) == pytest.approx([1.0, -1.0])


class TestNoLookahead:
    """The signals module's contract: data after t cannot move the signal at t."""

    def test_truncation_invariance(self):
        returns = random_returns()
        full = time_series_momentum(returns, lookback=20)
        for cutoff in (25, 100, 250, 499):
            truncated = time_series_momentum(returns.iloc[:cutoff], lookback=20)
            pd.testing.assert_series_equal(full.iloc[:cutoff], truncated)

    def test_changing_the_future_does_not_change_the_past(self):
        returns = random_returns()
        altered = returns.copy()
        altered.iloc[400:] = -0.05  # rewrite the future
        original = time_series_momentum(returns, lookback=20)
        rerun = time_series_momentum(altered, lookback=20)
        pd.testing.assert_series_equal(original.iloc[:400], rerun.iloc[:400])


class TestEndToEnd:
    def test_feeds_directly_into_engine(self):
        # The whole pipeline: returns -> signal -> lagged positions -> net P&L,
        # with realistic costs. Smoke test plus sanity on the outputs.
        returns = random_returns()
        signal = time_series_momentum(returns, lookback=63)
        result = run_backtest(returns, signal, BpsCost())
        report = result.summary()
        assert set(signal.unique()) <= {-1.0, 0.0, 1.0}
        assert report["n_periods"] == len(returns)
        assert report["max_drawdown"] <= 0.0
        assert report["turnover"] >= 0.0
