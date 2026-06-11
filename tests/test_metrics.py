"""Unit tests for backtester.metrics.

Expected values are computed by hand (shown in comments) rather than by
re-running the library code, so a bug in the implementation cannot hide
inside the test.
"""

import numpy as np
import pandas as pd
import pytest

from backtester.metrics import (
    annualized_return,
    annualized_volatility,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
    summary,
    turnover,
)


def series(*values: float) -> pd.Series:
    return pd.Series(list(values), dtype=np.float64)


class TestValidation:
    def test_empty_series_raises(self):
        with pytest.raises(ValueError, match="empty"):
            sharpe_ratio(pd.Series(dtype=np.float64))

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="NaN"):
            max_drawdown(series(0.01, np.nan, 0.02))

    def test_non_series_raises(self):
        with pytest.raises(TypeError, match="pandas Series"):
            annualized_return([0.01, 0.02])


class TestAnnualizedReturn:
    def test_compounds_geometrically(self):
        # (1.10 * 0.95) ** (252 / 2) - 1 over two periods of one "year" each:
        # use periods_per_year=2 so the two periods are exactly one year.
        # total growth = 1.10 * 0.95 = 1.045 -> CAGR = 1.045^(2/2) - 1 = 0.045
        result = annualized_return(series(0.10, -0.05), periods_per_year=2)
        assert result == pytest.approx(0.045)

    def test_flat_returns_zero(self):
        assert annualized_return(series(0.0, 0.0, 0.0)) == pytest.approx(0.0)

    def test_total_loss_is_nan(self):
        # A -100% period wipes out the account; CAGR is undefined.
        assert np.isnan(annualized_return(series(0.05, -1.0)))


class TestAnnualizedVolatility:
    def test_matches_hand_computation(self):
        # returns 0.01, 0.03: mean 0.02, sample std = sqrt(((−0.01)^2 + 0.01^2)/1)
        # = sqrt(0.0002) ≈ 0.0141421; annualized with 4 periods/yr: * sqrt(4) = 2x
        result = annualized_volatility(series(0.01, 0.03), periods_per_year=4)
        assert result == pytest.approx(np.sqrt(0.0002) * 2.0)

    def test_single_observation_is_nan(self):
        assert np.isnan(annualized_volatility(series(0.01)))


class TestSharpeRatio:
    def test_matches_hand_computation(self):
        # returns 0.01, 0.02, 0.03 with rf = 0:
        # mean = 0.02, sample std = 0.01, periods_per_year = 252
        # sharpe = 0.02 / 0.01 * sqrt(252) = 2 * sqrt(252)
        result = sharpe_ratio(series(0.01, 0.02, 0.03))
        assert result == pytest.approx(2.0 * np.sqrt(252))

    def test_risk_free_rate_lowers_sharpe(self):
        rets = series(0.01, 0.02, 0.03)
        assert sharpe_ratio(rets, risk_free_rate=0.05) < sharpe_ratio(rets)

    def test_zero_volatility_is_nan(self):
        assert np.isnan(sharpe_ratio(series(0.01, 0.01, 0.01)))

    def test_negative_mean_gives_negative_sharpe(self):
        assert sharpe_ratio(series(-0.01, -0.02, -0.03)) < 0.0


class TestMaxDrawdown:
    def test_known_path(self):
        # equity: 1.10 -> 0.88 (peak 1.10, trough 0.88) -> dd = 0.88/1.10 - 1 = -0.20
        result = max_drawdown(series(0.10, -0.20))
        assert result == pytest.approx(-0.20)

    def test_monotonic_gains_have_zero_drawdown(self):
        assert max_drawdown(series(0.01, 0.02, 0.03)) == pytest.approx(0.0)

    def test_drawdown_measured_from_running_peak(self):
        # equity: 1.20, 0.96, 1.152 — recovery does not erase the -20% drawdown
        result = max_drawdown(series(0.20, -0.20, 0.20))
        assert result == pytest.approx(-0.20)

    def test_is_never_positive(self):
        rng = np.random.default_rng(7)
        rets = pd.Series(rng.normal(0.001, 0.02, size=500))
        assert max_drawdown(rets) <= 0.0


class TestHitRate:
    def test_counts_only_active_periods(self):
        # 2 wins, 1 loss, 2 flat -> 2 / 3
        result = hit_rate(series(0.01, -0.01, 0.0, 0.02, 0.0))
        assert result == pytest.approx(2.0 / 3.0)

    def test_all_flat_is_nan(self):
        assert np.isnan(hit_rate(series(0.0, 0.0)))


class TestTurnover:
    def test_initial_entry_counts_as_trade(self):
        # weights 1, 1, 1: one trade of size 1 over 3 periods
        # mean |trade| = 1/3, annualized at 3 periods/yr -> 1.0 (one-way / year)
        result = turnover(series(1.0, 1.0, 1.0), periods_per_year=3)
        assert result == pytest.approx(1.0)

    def test_flip_long_to_short(self):
        # weights 1, -1: trades |1| + |-2| = 3 over 2 periods
        # mean = 1.5, annualized at 2 periods/yr -> 3.0
        result = turnover(series(1.0, -1.0), periods_per_year=2)
        assert result == pytest.approx(3.0)

    def test_never_trading_from_flat_is_zero(self):
        assert turnover(series(0.0, 0.0, 0.0)) == pytest.approx(0.0)


class TestSummary:
    def test_contains_standard_metrics(self):
        result = summary(series(0.01, -0.02, 0.03))
        expected_keys = {
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "hit_rate",
            "n_periods",
        }
        assert set(result) == expected_keys
        assert result["n_periods"] == 3.0

    def test_includes_turnover_with_positions(self):
        result = summary(series(0.01, 0.02), positions=series(1.0, 0.0))
        assert "turnover" in result

    def test_values_match_individual_functions(self):
        rets = series(0.01, -0.02, 0.03, 0.005)
        result = summary(rets)
        assert result["sharpe_ratio"] == pytest.approx(sharpe_ratio(rets))
        assert result["max_drawdown"] == pytest.approx(max_drawdown(rets))
