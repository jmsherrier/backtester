"""Unit and integration tests for backtester.engine.

Beyond hand-computed unit cases, this file carries the two integration
guarantees the README promises:

1. No lookahead: a signal that "knows" the same-day return makes money in a
   broken (unlagged) engine every single period; under the real engine its
   edge must disappear.
2. Cost reconciliation: net + costs == gross, exactly, every period.
"""

import numpy as np
import pandas as pd
import pytest

from backtester.engine import run_backtest
from backtester.execution import BpsCost, ZeroCost
from backtester.metrics import sharpe_ratio


def series(*values: float) -> pd.Series:
    return pd.Series(list(values), dtype=np.float64)


def random_returns(n: int = 1000, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0, 0.01, size=n))


class TestValidation:
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            run_backtest(pd.Series(dtype=np.float64), pd.Series(dtype=np.float64))

    def test_nan_signal_raises(self):
        with pytest.raises(ValueError, match="NaN"):
            run_backtest(series(0.01, 0.02), series(np.nan, 1.0))

    def test_mismatched_index_raises(self):
        returns = pd.Series([0.01, 0.02], index=[0, 1])
        signal = pd.Series([1.0, 1.0], index=[1, 2])
        with pytest.raises(ValueError, match="identical index"):
            run_backtest(returns, signal)

    def test_non_series_raises(self):
        with pytest.raises(TypeError, match="pandas Series"):
            run_backtest([0.01], series(1.0))


class TestLag:
    def test_signal_earns_next_period_return(self):
        # signal fires only on day 0; the position is held during day 1,
        # so the strategy earns exactly day 1's return and nothing else.
        returns = series(0.10, 0.20, 0.30)
        signal = series(1.0, 0.0, 0.0)
        result = run_backtest(returns, signal)
        assert list(result.positions) == pytest.approx([0.0, 1.0, 0.0])
        assert list(result.gross_returns) == pytest.approx([0.0, 0.20, 0.0])

    def test_first_period_is_always_flat(self):
        result = run_backtest(series(0.10, 0.10), series(1.0, 1.0))
        assert result.positions.iloc[0] == 0.0
        assert result.gross_returns.iloc[0] == 0.0

    def test_last_signal_value_is_never_traded_on(self):
        # A huge signal on the final date has nothing left to earn.
        returns = series(0.01, 0.01)
        with_final_signal = run_backtest(returns, series(0.0, 1.0))
        without = run_backtest(returns, series(0.0, 0.0))
        assert list(with_final_signal.gross_returns) == list(without.gross_returns)


class TestNoLookahead:
    """The README's core promise, as an executable test."""

    def test_same_day_oracle_signal_has_no_edge(self):
        # The cheating signal: sign of the SAME day's return. An engine that
        # forgets to lag would earn |r_t| > 0 every period — a money machine.
        # The real engine holds sign(r_{t-1}) during t, which on iid returns
        # is a coin flip.
        returns = random_returns()
        oracle = np.sign(returns)

        broken_engine_gross = (oracle * returns).mean()  # what forgetting the lag pays
        result = run_backtest(returns, oracle, ZeroCost())
        honest_gross = result.gross_returns.mean()

        # The unlagged engine would earn the mean absolute return — around
        # 80 bps/day here. The honest engine should be within noise of zero.
        assert broken_engine_gross == pytest.approx(returns.abs().mean())
        assert abs(honest_gross) < 0.1 * broken_engine_gross

    def test_oracle_sharpe_is_not_too_good_to_be_true(self):
        returns = random_returns(seed=7)
        result = run_backtest(returns, np.sign(returns), ZeroCost())
        # Unlagged, this strategy's Sharpe is astronomical (vol of |r| is tiny
        # relative to its mean). Lagged, it must look like luck at best.
        assert abs(sharpe_ratio(result.net_returns)) < 1.0


class TestCostAccounting:
    def test_net_plus_costs_reconciles_to_gross_exactly(self):
        returns = random_returns(seed=3)
        rng = np.random.default_rng(4)
        signal = pd.Series(rng.choice([-1.0, 0.0, 1.0], size=len(returns)))
        result = run_backtest(returns, signal, BpsCost())
        # (gross - costs) + costs re-rounds, so the identity holds to float
        # precision (~1 ulp), not bit-exactly. Anything beyond that would be
        # a real accounting bug, so the tolerance is machine epsilon, not 1e-8.
        residual = result.net_returns + result.costs - result.gross_returns
        assert residual.abs().max() < np.finfo(np.float64).eps

    def test_costs_match_hand_computation(self):
        # signal 1, 1, 0 lags to positions 0, 1, 1: one entry trade of +1 on
        # day 1, then holding. At the default 6 bps total rate the only cost
        # is 1.0 * 0.0006 on day 1.
        result = run_backtest(series(0.0, 0.0, 0.0), series(1.0, 1.0, 0.0), BpsCost())
        assert list(result.costs) == pytest.approx([0.0, 0.0006, 0.0])

    def test_zero_cost_means_net_equals_gross(self):
        returns = random_returns(seed=5)
        signal = pd.Series(np.ones(len(returns)))
        result = run_backtest(returns, signal, ZeroCost())
        assert result.net_returns.equals(result.gross_returns)

    def test_default_cost_model_is_zero_cost(self):
        returns = series(0.01, 0.02)
        explicit = run_backtest(returns, series(1.0, 1.0), ZeroCost())
        default = run_backtest(returns, series(1.0, 1.0))
        assert default.net_returns.equals(explicit.net_returns)


class TestBuyAndHold:
    def test_full_period_long(self):
        # signal always 1 -> in the market from day 1 onward (day 0 flat),
        # paying one entry trade. Hand-check every series.
        returns = series(0.10, -0.05, 0.02)
        result = run_backtest(returns, series(1.0, 1.0, 1.0), BpsCost())
        assert list(result.positions) == pytest.approx([0.0, 1.0, 1.0])
        assert list(result.trades) == pytest.approx([0.0, 1.0, 0.0])
        assert list(result.gross_returns) == pytest.approx([0.0, -0.05, 0.02])
        assert list(result.costs) == pytest.approx([0.0, 0.0006, 0.0])
        assert list(result.net_returns) == pytest.approx([0.0, -0.0506, 0.02])


class TestResultSummary:
    def test_summary_reports_net_metrics_with_turnover(self):
        returns = random_returns(seed=9)
        signal = pd.Series(np.sign(np.sin(np.arange(len(returns)))))
        result = run_backtest(returns, signal, BpsCost())
        report = result.summary()
        assert "turnover" in report
        assert report["sharpe_ratio"] == pytest.approx(sharpe_ratio(result.net_returns))

    def test_costs_lower_the_summary_sharpe(self):
        returns = random_returns(seed=10) + 0.0005  # give the strategy an edge
        signal = pd.Series(np.ones(len(returns)))
        gross_sharpe = run_backtest(returns, signal, ZeroCost()).summary()["sharpe_ratio"]
        net_sharpe = run_backtest(returns, signal, BpsCost()).summary()["sharpe_ratio"]
        assert net_sharpe < gross_sharpe
