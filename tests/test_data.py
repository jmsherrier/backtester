"""Unit tests for backtester.data.

The loader's contract is reject-don't-repair: every test in TestLoaderRejects
pins down a failure mode that quieter libraries would paper over (forward-
fill, keep-first dedup, NaN propagation).
"""

import numpy as np
import pandas as pd
import pytest

from backtester.data import generate_gbm_prices, load_prices_csv, prices_to_returns


def write_csv(tmp_path, text: str, name: str = "prices.csv"):
    path = tmp_path / name
    path.write_text(text)
    return path


GOOD_CSV = """Date,Close
2024-01-02,100.0
2024-01-03,101.0
2024-01-04,99.5
"""


class TestLoadPricesCsv:
    def test_loads_sorted_datetime_indexed_series(self, tmp_path):
        prices = load_prices_csv(write_csv(tmp_path, GOOD_CSV))
        assert isinstance(prices.index, pd.DatetimeIndex)
        assert list(prices) == pytest.approx([100.0, 101.0, 99.5])
        assert prices.index[0] == pd.Timestamp("2024-01-02")

    def test_out_of_order_rows_are_sorted_not_rejected(self, tmp_path):
        csv = "Date,Close\n2024-01-04,99.5\n2024-01-02,100.0\n2024-01-03,101.0\n"
        prices = load_prices_csv(write_csv(tmp_path, csv))
        assert prices.index.is_monotonic_increasing
        assert prices.iloc[0] == 100.0

    def test_custom_column_names(self, tmp_path):
        csv = "timestamp,adj_close\n2024-01-02,50.0\n2024-01-03,51.0\n"
        prices = load_prices_csv(
            write_csv(tmp_path, csv), date_column="timestamp", price_column="adj_close"
        )
        assert len(prices) == 2


class TestLoaderRejects:
    def test_missing_column(self, tmp_path):
        csv = "Date,Open\n2024-01-02,100.0\n"
        with pytest.raises(ValueError, match="missing column 'Close'"):
            load_prices_csv(write_csv(tmp_path, csv))

    def test_duplicate_dates(self, tmp_path):
        csv = "Date,Close\n2024-01-02,100.0\n2024-01-02,101.0\n"
        with pytest.raises(ValueError, match="duplicate dates"):
            load_prices_csv(write_csv(tmp_path, csv))

    def test_missing_price_is_not_forward_filled(self, tmp_path):
        csv = "Date,Close\n2024-01-02,100.0\n2024-01-03,\n2024-01-04,99.0\n"
        with pytest.raises(ValueError, match="does not forward-fill"):
            load_prices_csv(write_csv(tmp_path, csv))

    def test_non_positive_price(self, tmp_path):
        csv = "Date,Close\n2024-01-02,100.0\n2024-01-03,-1.0\n"
        with pytest.raises(ValueError, match="non-positive"):
            load_prices_csv(write_csv(tmp_path, csv))


class TestPricesToReturns:
    def test_matches_hand_computation(self):
        # 100 -> 110 is +10%; 110 -> 99 is -10%
        prices = pd.Series([100.0, 110.0, 99.0])
        result = prices_to_returns(prices)
        assert list(result) == pytest.approx([0.10, -0.10])

    def test_first_period_dropped_not_nan(self):
        result = prices_to_returns(pd.Series([100.0, 101.0]))
        assert len(result) == 1
        assert not result.isna().any()

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            prices_to_returns(pd.Series([100.0]))

    def test_round_trips_with_gbm_generator(self):
        prices = generate_gbm_prices(n_periods=300, seed=5)
        returns = prices_to_returns(prices)
        rebuilt = prices.iloc[0] * (1 + returns).cumprod()
        pd.testing.assert_series_equal(rebuilt, prices.iloc[1:], check_names=False)


class TestGenerateGbmPrices:
    def test_reproducible_with_seed(self):
        a = generate_gbm_prices(seed=11)
        b = generate_gbm_prices(seed=11)
        pd.testing.assert_series_equal(a, b)

    def test_different_seeds_differ(self):
        assert not generate_gbm_prices(seed=1).equals(generate_gbm_prices(seed=2))

    def test_starts_at_initial_price_and_stays_positive(self):
        prices = generate_gbm_prices(initial_price=42.0, seed=3)
        assert prices.iloc[0] == 42.0
        assert (prices > 0).all()

    def test_business_day_index(self):
        prices = generate_gbm_prices(n_periods=10)
        assert prices.index.dayofweek.max() <= 4  # no weekends

    def test_realized_volatility_near_target(self):
        # 20% annual vol over ~8 years of daily data should realize within
        # a few percent of target (chi-square concentration).
        prices = generate_gbm_prices(n_periods=2000, annual_volatility=0.20, seed=8)
        realized = prices_to_returns(prices).std(ddof=1) * np.sqrt(252)
        assert realized == pytest.approx(0.20, rel=0.10)

    def test_bad_params_raise(self):
        with pytest.raises(ValueError, match="n_periods"):
            generate_gbm_prices(n_periods=1)
        with pytest.raises(ValueError, match="initial_price"):
            generate_gbm_prices(initial_price=0.0)
