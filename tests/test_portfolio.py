"""Unit and integration tests for the multi-asset engine.

The single-asset engine's two guarantees, re-pinned at the portfolio level:
no lookahead (the per-asset lag still holds when summed across a book) and
cost reconciliation (net + costs == gross, exactly, every period). Plus the
multi-asset-specific behavior: per-asset marking summed into one return
series, and a dollar-neutral book netting its legs.
"""

import numpy as np
import pandas as pd
import pytest

from backtester.data import generate_gbm_panel, prices_to_returns
from backtester.engine import run_portfolio_backtest
from backtester.execution import BpsCost, ZeroCost
from backtester.metrics import sharpe_ratio
from backtester.signals import cross_sectional_momentum


def frame(data: dict, index=None) -> pd.DataFrame:
    return pd.DataFrame(data, index=index, dtype=np.float64)


def panel(n_assets: int = 5, n_periods: int = 500, seed: int = 1) -> pd.DataFrame:
    return prices_to_returns(
        generate_gbm_panel(n_assets=n_assets, n_periods=n_periods, seed=seed)
    )


class TestValidation:
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            run_portfolio_backtest(pd.DataFrame(), pd.DataFrame())

    def test_nan_signal_raises(self):
        returns = frame({"A": [0.01, 0.02], "B": [0.0, 0.01]})
        signal = frame({"A": [np.nan, 1.0], "B": [0.0, 0.0]})
        with pytest.raises(ValueError, match="NaN"):
            run_portfolio_backtest(returns, signal)

    def test_non_dataframe_raises(self):
        with pytest.raises(TypeError, match="DataFrame"):
            run_portfolio_backtest([0.01], frame({"A": [1.0]}))

    def test_mismatched_index_raises(self):
        returns = frame({"A": [0.01, 0.02]}, index=[0, 1])
        signal = frame({"A": [1.0, 1.0]}, index=[1, 2])
        with pytest.raises(ValueError, match="identical index"):
            run_portfolio_backtest(returns, signal)

    def test_mismatched_columns_raise(self):
        returns = frame({"A": [0.01, 0.02], "B": [0.0, 0.0]})
        signal = frame({"A": [1.0, 1.0], "C": [0.0, 0.0]})
        with pytest.raises(ValueError, match="identical columns"):
            run_portfolio_backtest(returns, signal)


class TestLagAndAggregation:
    def test_positions_are_lagged_per_asset_and_book_starts_flat(self):
        returns = frame({"A": [0.10, 0.20, 0.30], "B": [0.0, 0.0, 0.0]})
        signal = frame({"A": [1.0, 0.0, 0.0], "B": [0.0, 0.0, 0.0]})
        result = run_portfolio_backtest(returns, signal)
        assert list(result.positions["A"]) == pytest.approx([0.0, 1.0, 0.0])
        assert result.positions.iloc[0].tolist() == pytest.approx([0.0, 0.0])

    def test_portfolio_gross_is_the_row_sum_of_asset_pnl(self):
        # A held long during day 1 earns +0.20; B short earns -(-0.10)=+0.10.
        returns = frame({"A": [0.0, 0.20, 0.0], "B": [0.0, -0.10, 0.0]})
        signal = frame({"A": [1.0, 1.0, 1.0], "B": [-1.0, -1.0, -1.0]})
        result = run_portfolio_backtest(returns, signal)
        assert result.gross_returns.iloc[1] == pytest.approx(0.30)

    def test_dollar_neutral_book_nets_its_legs(self):
        # Two assets with identical returns, held +1 and -1: the legs cancel
        # exactly, so portfolio gross is zero every period despite moves.
        returns = frame({"A": [0.05, -0.03, 0.02], "B": [0.05, -0.03, 0.02]})
        signal = frame({"A": [1.0, 1.0, 1.0], "B": [-1.0, -1.0, -1.0]})
        result = run_portfolio_backtest(returns, signal)
        assert np.allclose(result.gross_returns.to_numpy(), 0.0)


class TestNoLookahead:
    def test_same_day_oracle_book_has_no_edge(self):
        # Per asset, the cheating signal is the sign of the same day's return;
        # an unlagged engine would earn the summed absolute returns. The real
        # engine lags every column, so on iid returns the edge vanishes.
        returns = panel(n_assets=5, n_periods=1000, seed=42)
        oracle = np.sign(returns)
        broken_gross = (oracle * returns).sum(axis=1).mean()
        honest_gross = run_portfolio_backtest(returns, oracle, ZeroCost()).gross_returns.mean()
        assert broken_gross == pytest.approx(returns.abs().sum(axis=1).mean())
        assert abs(honest_gross) < 0.1 * broken_gross


class TestCostAccounting:
    def test_net_plus_costs_reconciles_to_gross_exactly(self):
        returns = panel(seed=3)
        rng = np.random.default_rng(4)
        signal = pd.DataFrame(
            rng.choice([-1.0, 0.0, 1.0], size=returns.shape),
            index=returns.index,
            columns=returns.columns,
        )
        result = run_portfolio_backtest(returns, signal, BpsCost())
        residual = result.net_returns + result.costs - result.gross_returns
        assert residual.abs().max() < np.finfo(np.float64).eps

    def test_costs_charge_each_asset_on_its_own_notional(self):
        # positions lag to A:0,1,1 and B:0,-1,-1 -> day 1 trades +1 and -1,
        # each 1.0 notional at the default 6 bps -> total 0.0012 on day 1. The
        # day-2 signal of 0 would only be traded on day 3 (outside the window),
        # so no exit cost lands here.
        returns = frame({"A": [0.0, 0.0, 0.0], "B": [0.0, 0.0, 0.0]})
        signal = frame({"A": [1.0, 1.0, 0.0], "B": [-1.0, -1.0, 0.0]})
        result = run_portfolio_backtest(returns, signal, BpsCost())
        assert list(result.costs) == pytest.approx([0.0, 0.0012, 0.0])

    def test_default_cost_model_is_zero_cost(self):
        returns = panel(seed=5)
        signal = pd.DataFrame(1.0, index=returns.index, columns=returns.columns)
        explicit = run_portfolio_backtest(returns, signal, ZeroCost())
        default = run_portfolio_backtest(returns, signal)
        assert default.net_returns.equals(explicit.net_returns)


class TestResultSummary:
    def test_summary_reports_net_metrics_with_book_turnover(self):
        returns = panel(seed=9)
        signal = cross_sectional_momentum(returns, lookback=63)
        result = run_portfolio_backtest(returns, signal, BpsCost())
        report = result.summary()
        assert report["sharpe_ratio"] == pytest.approx(sharpe_ratio(result.net_returns))
        # Book turnover sums both legs: the manual annualized figure must match.
        expected = result.trades.abs().sum(axis=1).mean() * 252
        assert report["turnover"] == pytest.approx(expected)

    def test_end_to_end_cross_sectional_on_independent_panel(self):
        # The honesty check: independent GBM columns have no cross-sectional
        # structure, so a winners-minus-losers book should show no real edge —
        # a net Sharpe within noise of zero, made worse by costs.
        returns = panel(n_assets=10, n_periods=2000, seed=0)
        signal = cross_sectional_momentum(returns, lookback=63)
        result = run_portfolio_backtest(returns, signal, BpsCost())
        report = result.summary()
        assert report["n_periods"] == len(returns)
        assert report["max_drawdown"] <= 0.0
        assert abs(report["sharpe_ratio"]) < 1.0
