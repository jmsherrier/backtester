"""Unit tests for backtester.execution.

Expected values are computed by hand (shown in comments), consistent with
the convention in test_metrics.py.
"""

import numpy as np
import pandas as pd
import pytest

from backtester.execution import BpsCost, CostModel, ZeroCost, trades_from_positions
from backtester.metrics import turnover


def series(*values: float) -> pd.Series:
    return pd.Series(list(values), dtype=np.float64)


class TestValidation:
    def test_empty_trades_raises(self):
        with pytest.raises(ValueError, match="empty"):
            BpsCost().cost(pd.Series(dtype=np.float64))

    def test_nan_trades_raises(self):
        with pytest.raises(ValueError, match="NaN"):
            ZeroCost().cost(series(0.5, np.nan))

    def test_non_series_raises(self):
        with pytest.raises(TypeError, match="pandas Series"):
            BpsCost().cost([0.5, -0.5])


class TestTradesFromPositions:
    def test_initial_entry_counts_as_trade(self):
        # weights 1, 1, 0.5: trades are 1 (entry), 0, -0.5
        result = trades_from_positions(series(1.0, 1.0, 0.5))
        assert list(result) == pytest.approx([1.0, 0.0, -0.5])

    def test_consistent_with_turnover_convention(self):
        # metrics.turnover uses the same enter-from-flat convention, so the
        # two must agree: turnover == mean(|trades|) * periods_per_year
        positions = series(1.0, -1.0, 0.0, 0.5)
        trades = trades_from_positions(positions)
        expected = float(trades.abs().mean()) * 252
        assert turnover(positions) == pytest.approx(expected)


class TestZeroCost:
    def test_always_zero(self):
        result = ZeroCost().cost(series(1.0, -2.0, 0.0))
        assert (result == 0.0).all()

    def test_preserves_index(self):
        idx = pd.date_range("2024-01-02", periods=3, freq="B")
        trades = pd.Series([1.0, 0.0, -1.0], index=idx)
        assert ZeroCost().cost(trades).index.equals(idx)


class TestBpsCost:
    def test_matches_hand_computation(self):
        # 1 bp commission + 5 bps slippage = 6 bps = 0.0006 per unit notional
        # trade of 0.5 weight -> cost 0.5 * 0.0006 = 0.0003
        result = BpsCost(commission_bps=1.0, slippage_bps=5.0).cost(series(0.5))
        assert result.iloc[0] == pytest.approx(0.0003)

    def test_sells_cost_the_same_as_buys(self):
        model = BpsCost()
        buys = model.cost(series(0.5))
        sells = model.cost(series(-0.5))
        assert buys.iloc[0] == pytest.approx(sells.iloc[0])

    def test_no_trade_costs_nothing(self):
        result = BpsCost().cost(series(0.0, 0.0))
        assert (result == 0.0).all()

    def test_costs_are_never_negative(self):
        rng = np.random.default_rng(11)
        trades = pd.Series(rng.normal(0.0, 1.0, size=200))
        assert (BpsCost().cost(trades) >= 0.0).all()

    def test_negative_rate_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            BpsCost(commission_bps=-1.0)

    def test_total_rate(self):
        assert BpsCost(commission_bps=2.0, slippage_bps=8.0).total_rate == pytest.approx(0.001)

    def test_is_a_cost_model(self):
        assert isinstance(BpsCost(), CostModel)
        assert isinstance(ZeroCost(), CostModel)


class TestCostDragEndToEnd:
    def test_round_trip_drag_on_flat_returns(self):
        # Enter full long then exit: trades 1.0 and -1.0 -> total cost
        # 2 * 6 bps = 12 bps of drag across the two periods.
        positions = series(1.0, 0.0)
        gross = series(0.0, 0.0)
        costs = BpsCost().cost(trades_from_positions(positions))
        net = gross - costs
        assert float(net.sum()) == pytest.approx(-0.0012)
