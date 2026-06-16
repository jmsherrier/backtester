"""Unit tests for backtester.data.panel.

The load-bearing property: a return matrix is a genuine rectangle. align_returns
refuses to invent values to square off assets with different histories (the
reject-don't-repair stance applied across assets), and common_window is the
explicit, visible way to take the shared date range instead.
"""

import numpy as np
import pandas as pd
import pytest

from backtester.data import (
    align_returns,
    common_window,
    generate_gbm_panel,
    load_price_panel,
    prices_to_returns,
)


def dated(values, start="2021-01-04") -> pd.Series:
    index = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=index, dtype=np.float64)


def write_price_csv(directory, ticker: str, rows: str) -> None:
    (directory / f"{ticker}.csv").write_text("Date,Close\n" + rows)


class TestAlignReturns:
    def test_aligns_shared_dates_into_columns(self):
        matrix = align_returns({"A": dated([0.01, 0.02]), "B": dated([-0.01, 0.0])})
        assert list(matrix.columns) == ["A", "B"]
        assert matrix.shape == (2, 2)
        assert matrix.loc[matrix.index[0], "B"] == pytest.approx(-0.01)

    def test_preserves_mapping_order(self):
        matrix = align_returns(
            {"Z": dated([0.0, 0.0]), "M": dated([0.0, 0.0]), "A": dated([0.0, 0.0])}
        )
        assert list(matrix.columns) == ["Z", "M", "A"]

    def test_empty_mapping_raises(self):
        with pytest.raises(ValueError, match="empty"):
            align_returns({})

    def test_ragged_dates_raise_with_offenders_named(self):
        # B is missing the third date A has -> ragged, must raise not fill.
        a = dated([0.01, 0.02, 0.03])
        b = dated([0.01, 0.02])
        with pytest.raises(ValueError, match="identical date index"):
            align_returns({"A": a, "B": b})

    def test_does_not_forward_fill_the_hole(self):
        # Confirm the failure is about the hole, not silently patched.
        a = dated([0.01, 0.02, 0.03])
        b = dated([0.01, 0.02])
        try:
            align_returns({"A": a, "B": b})
        except ValueError as err:
            assert "forward-fill" in str(err)

    def test_rejects_nan_input(self):
        with pytest.raises(ValueError, match="NaN"):
            align_returns({"A": dated([0.01, np.nan])})

    def test_rejects_unsorted_input(self):
        a = dated([0.01, 0.02, 0.03]).iloc[::-1]
        with pytest.raises(ValueError, match="sorted"):
            align_returns({"A": a})

    def test_rejects_duplicate_dates(self):
        a = pd.Series([0.01, 0.02], index=pd.to_datetime(["2021-01-04"] * 2))
        with pytest.raises(ValueError, match="duplicate"):
            align_returns({"A": a})


class TestCommonWindow:
    def test_takes_the_intersection(self):
        a = dated([0.01, 0.02, 0.03], start="2021-01-04")  # Mon-Wed
        b = dated([0.05, 0.06, 0.07], start="2021-01-05")  # Tue-Thu
        matrix = common_window({"A": a, "B": b})
        # Shared dates are Tue and Wed.
        assert len(matrix) == 2
        assert matrix.index[0] == pd.Timestamp("2021-01-05")
        assert matrix.loc[matrix.index[0], "A"] == pytest.approx(0.02)

    def test_no_overlap_raises(self):
        a = dated([0.01, 0.02], start="2021-01-04")
        b = dated([0.01, 0.02], start="2022-01-04")
        with pytest.raises(ValueError, match="no common dates"):
            common_window({"A": a, "B": b})

    def test_empty_mapping_raises(self):
        with pytest.raises(ValueError, match="empty"):
            common_window({})


class TestGenerateGbmPanel:
    def test_shape_and_naming(self):
        panel = generate_gbm_panel(n_assets=4, n_periods=50, seed=1)
        assert panel.shape == (50, 4)
        assert list(panel.columns) == ["ASSET_00", "ASSET_01", "ASSET_02", "ASSET_03"]

    def test_reproducible_with_seed(self):
        pd.testing.assert_frame_equal(
            generate_gbm_panel(seed=7), generate_gbm_panel(seed=7)
        )

    def test_all_columns_start_at_100_and_stay_positive(self):
        panel = generate_gbm_panel(n_assets=3, n_periods=200, seed=2)
        assert (panel.iloc[0] == 100.0).all()
        assert (panel > 0).to_numpy().all()

    def test_columns_are_independent(self):
        # Independent GBM columns should be close to uncorrelated over a long
        # sample — well short of the near-1 a duplicated-draw bug would show.
        returns = prices_to_returns(generate_gbm_panel(n_assets=2, n_periods=3000, seed=4))
        corr = returns.corr().iloc[0, 1]
        assert abs(corr) < 0.1

    def test_bad_params_raise(self):
        with pytest.raises(ValueError, match="n_assets"):
            generate_gbm_panel(n_assets=0)
        with pytest.raises(ValueError, match="n_periods"):
            generate_gbm_panel(n_periods=1)


class TestPricesToReturnsPanel:
    def test_dataframe_returns_per_column(self):
        prices = pd.DataFrame(
            {"A": [100.0, 110.0, 99.0], "B": [50.0, 55.0, 55.0]},
            index=pd.bdate_range("2021-01-04", periods=3),
        )
        returns = prices_to_returns(prices)
        assert returns.shape == (2, 2)
        assert returns["A"].tolist() == pytest.approx([0.10, -0.10])
        assert returns["B"].tolist() == pytest.approx([0.10, 0.0])

    def test_first_row_dropped_not_nan(self):
        returns = prices_to_returns(generate_gbm_panel(n_assets=3, n_periods=10, seed=1))
        assert len(returns) == 9
        assert not returns.isna().to_numpy().any()

    def test_panel_round_trips_through_alignment(self):
        # Generate a panel, convert to returns, align -> recovers the matrix.
        panel = generate_gbm_panel(n_assets=3, n_periods=100, seed=9)
        returns = prices_to_returns(panel)
        aligned = align_returns({c: returns[c] for c in returns.columns})
        pd.testing.assert_frame_equal(aligned, returns)

    def test_dataframe_rejects_nan(self):
        prices = pd.DataFrame({"A": [100.0, np.nan, 99.0]})
        with pytest.raises(ValueError, match="NaN"):
            prices_to_returns(prices)

    def test_dataframe_rejects_non_positive(self):
        prices = pd.DataFrame({"A": [100.0, -1.0]})
        with pytest.raises(ValueError, match="positive"):
            prices_to_returns(prices)


class TestLoadPricePanel:
    def test_loads_each_file_keyed_by_stem(self, tmp_path):
        write_price_csv(tmp_path, "AAPL", "2024-01-02,100\n2024-01-03,101\n")
        write_price_csv(tmp_path, "MSFT", "2024-01-02,50\n2024-01-03,49\n")
        panel = load_price_panel(tmp_path)
        assert set(panel) == {"AAPL", "MSFT"}
        assert isinstance(panel["AAPL"], pd.Series)
        assert panel["AAPL"].iloc[-1] == 101.0

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not a directory"):
            load_price_panel(tmp_path / "nope")

    def test_no_matching_files_raises(self, tmp_path):
        with pytest.raises(ValueError, match="no files matching"):
            load_price_panel(tmp_path)

    def test_one_bad_file_fails_loudly(self, tmp_path):
        write_price_csv(tmp_path, "GOOD", "2024-01-02,100\n2024-01-03,101\n")
        write_price_csv(tmp_path, "BAD", "2024-01-02,100\n2024-01-02,101\n")  # dup date
        with pytest.raises(ValueError, match="duplicate dates"):
            load_price_panel(tmp_path)

    def test_composes_into_a_return_matrix(self, tmp_path):
        # The intended real-data path: load -> per-asset returns -> common_window.
        write_price_csv(tmp_path, "A", "2024-01-02,100\n2024-01-03,110\n2024-01-04,99\n")
        write_price_csv(tmp_path, "B", "2024-01-02,50\n2024-01-03,55\n2024-01-04,55\n")
        prices = load_price_panel(tmp_path)
        returns = common_window({t: prices_to_returns(p) for t, p in prices.items()})
        assert list(returns.columns) == ["A", "B"]
        assert returns.shape == (2, 2)
        assert returns["A"].tolist() == pytest.approx([0.10, -0.10])
