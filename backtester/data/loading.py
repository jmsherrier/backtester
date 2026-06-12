"""Price loading and return construction.

The policy throughout: reject suspect data loudly rather than repair it
silently. No forward-filling, no NaN-dropping, no auto-deduplication — a
gap or a duplicate in a price file is information about data quality, and
papering over it here would let it leak into every downstream result.

Raw market data files are intentionally gitignored; the synthetic generator
exists so examples and tests run from a clean clone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_prices_csv(
    path: str | Path,
    date_column: str = "Date",
    price_column: str = "Close",
) -> pd.Series:
    """Load a single asset's price series from a CSV file.

    Expects one row per period with a parseable date column and a price
    column (adjusted close, if you have it — unadjusted prices make every
    dividend look like a crash).

    Validation is strict and the failure modes are deliberate:

    - missing columns, unparseable dates -> raise
    - duplicate dates -> raise (no silent keep-first)
    - NaN or non-positive prices -> raise (no forward-fill)

    Rows are sorted by date before validation, so out-of-order files load
    fine; only genuinely bad data fails.
    """
    path = Path(path)
    frame = pd.read_csv(path)
    for column in (date_column, price_column):
        if column not in frame.columns:
            raise ValueError(
                f"{path.name}: missing column '{column}' "
                f"(found: {', '.join(frame.columns)})"
            )

    dates = pd.to_datetime(frame[date_column])  # raises on unparseable dates
    prices = pd.Series(
        frame[price_column].to_numpy(dtype=np.float64),
        index=pd.DatetimeIndex(dates),
        name=price_column,
    ).sort_index()

    if prices.index.has_duplicates:
        dupes = prices.index[prices.index.duplicated()].unique()
        raise ValueError(f"{path.name}: duplicate dates: {list(dupes[:5])}")
    if prices.isna().any():
        raise ValueError(
            f"{path.name}: {int(prices.isna().sum())} missing price(s); "
            "this loader does not forward-fill"
        )
    if (prices <= 0).any():
        raise ValueError(f"{path.name}: non-positive prices found")
    return prices


def prices_to_returns(prices: pd.Series) -> pd.Series:
    """Simple periodic returns from a price series.

    The first period has no prior price and is dropped, not NaN-filled —
    the result is ready for the engine, which rejects NaN by design.
    """
    if not isinstance(prices, pd.Series):
        raise TypeError(f"prices must be a pandas Series, got {type(prices).__name__}")
    if len(prices) < 2:
        raise ValueError("need at least 2 prices to compute a return")
    if prices.isna().any():
        raise ValueError("prices contains NaN value(s)")
    if (prices <= 0).any():
        raise ValueError("prices must be strictly positive")
    return prices.astype(np.float64).pct_change().iloc[1:]


def generate_gbm_prices(
    n_periods: int = 1000,
    annual_drift: float = 0.05,
    annual_volatility: float = 0.20,
    initial_price: float = 100.0,
    periods_per_year: int = 252,
    seed: int = 0,
) -> pd.Series:
    """Synthetic daily prices from geometric Brownian motion.

    For examples and tests only, and useful precisely because GBM has *no
    exploitable structure*: returns are independent, so any strategy showing
    an edge on this data is exposing a bug (most likely lookahead) rather
    than finding alpha. Seeded for reproducibility.

    The index is business days starting 2020-01-01 — synthetic dates for a
    synthetic asset, but calendar-shaped so examples look like real usage.
    """
    if n_periods < 2:
        raise ValueError(f"n_periods must be >= 2, got {n_periods}")
    if annual_volatility < 0:
        raise ValueError(f"annual_volatility must be >= 0, got {annual_volatility}")
    if initial_price <= 0:
        raise ValueError(f"initial_price must be > 0, got {initial_price}")

    rng = np.random.default_rng(seed)
    dt = 1.0 / periods_per_year
    # Exact GBM discretization: log-returns are iid normal with these moments.
    log_returns = rng.normal(
        loc=(annual_drift - 0.5 * annual_volatility**2) * dt,
        scale=annual_volatility * np.sqrt(dt),
        size=n_periods - 1,
    )
    log_path = np.concatenate([[0.0], np.cumsum(log_returns)])
    index = pd.bdate_range("2020-01-01", periods=n_periods)
    return pd.Series(initial_price * np.exp(log_path), index=index, name="Close")
