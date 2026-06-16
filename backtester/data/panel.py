"""Multi-asset return matrices.

A panel is a rectangular table of returns: one row per date, one column per
asset, no holes. Everything cross-sectional downstream (ranking assets against
each other, sizing a long-short book) assumes that rectangle is real — that on
any given row, every asset's number is a genuine observation for that date.

So the alignment policy is the same reject-don't-repair stance as the single-
asset loader, applied to the seam *between* assets: if two assets do not cover
exactly the same dates, that raggedness is information, not something to patch.
:func:`align_returns` refuses to invent values to square off the rectangle, and
names the offending dates so you can decide explicitly. The deliberate, visible
way to get a clean panel from assets with different histories is
:func:`common_window`, which intersects the dates and tells you what it dropped.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from backtester.data.loading import load_prices_csv


def load_price_panel(
    directory: str | Path,
    pattern: str = "*.csv",
    date_column: str = "Date",
    price_column: str = "Close",
) -> dict[str, pd.Series]:
    """Strict-load every price CSV in a directory, keyed by file stem (ticker).

    Each file is loaded by :func:`backtester.data.loading.load_prices_csv`, so
    the same per-file contract holds — parseable dates, no duplicate dates, no
    NaN, strictly positive prices — and a single bad file fails loudly rather
    than quietly dropping an asset.

    Deliberately stops at loading: it returns one price ``Series`` per ticker
    and does *not* align them. Assets list and delist at different times, and
    forcing a rectangle is a decision with consequences (drop dates? whose
    calendar?), so alignment stays an explicit caller step — convert each
    series with :func:`prices_to_returns` and combine with :func:`align_returns`
    (strict) or :func:`common_window` (shared-date intersection).

    Parameters
    ----------
    directory : str | Path
        Folder of price CSVs; each file's stem becomes its ticker (so
        ``AAPL.csv`` -> ``"AAPL"``).
    pattern : str
        Glob for the files to include (default ``"*.csv"``).
    date_column, price_column : str
        Column names passed through to ``load_prices_csv``.

    Returns
    -------
    dict[str, pd.Series]
        Ticker -> price series, sorted by ticker for a stable column order.

    Raises
    ------
    ValueError
        If ``directory`` is not a directory, or no file matches ``pattern``.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"not a directory: {directory}")
    paths = sorted(directory.glob(pattern))
    if not paths:
        raise ValueError(f"no files matching '{pattern}' in {directory}")
    return {
        path.stem: load_prices_csv(path, date_column, price_column) for path in paths
    }


def _validate_asset_series(returns: pd.Series, asset: str) -> pd.Series:
    if not isinstance(returns, pd.Series):
        raise TypeError(
            f"asset '{asset}': expected a pandas Series, got {type(returns).__name__}"
        )
    if returns.empty:
        raise ValueError(f"asset '{asset}': series is empty")
    if returns.isna().any():
        raise ValueError(
            f"asset '{asset}': contains {int(returns.isna().sum())} NaN value(s) "
            "before alignment; clean or investigate upstream"
        )
    if not returns.index.is_monotonic_increasing:
        raise ValueError(f"asset '{asset}': index must be sorted ascending")
    if returns.index.has_duplicates:
        dupes = returns.index[returns.index.duplicated()].unique()
        raise ValueError(f"asset '{asset}': duplicate dates: {list(dupes[:5])}")
    return returns.astype(np.float64)


def align_returns(returns_by_asset: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Assemble per-asset return series into one aligned return matrix.

    Parameters
    ----------
    returns_by_asset : Mapping[str, pd.Series]
        Asset label -> periodic return series. Each series is validated the
        same way the engine validates its inputs (non-empty, sorted, unique
        index, no NaN).

    Returns
    -------
    pd.DataFrame
        Float64 returns, one column per asset (in the mapping's order), indexed
        by the shared dates.

    Raises
    ------
    ValueError
        If the mapping is empty, or if the assets do not all cover exactly the
        same dates. A ragged panel is rejected rather than forward-filled or
        silently inner-joined — the error names the dates that are missing for
        some asset so the gap can be resolved on purpose (e.g. via
        :func:`common_window`).
    """
    if not returns_by_asset:
        raise ValueError("returns_by_asset is empty")

    columns = {
        asset: _validate_asset_series(series, asset)
        for asset, series in returns_by_asset.items()
    }
    matrix = pd.concat(columns, axis=1)  # outer join: holes become NaN

    if matrix.isna().any().to_numpy().any():
        ragged = matrix.index[matrix.isna().any(axis=1)]
        raise ValueError(
            f"assets do not share an identical date index: {len(ragged)} date(s) "
            f"are present for some assets but not others (e.g. "
            f"{[str(d.date()) for d in ragged[:5]]}). This loader does not "
            "forward-fill or inner-join silently; use common_window() to take "
            "the shared date range on purpose."
        )
    return matrix


def common_window(returns_by_asset: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Align on the dates common to *every* asset, dropping the rest.

    This is the explicit, auditable inner join: the parts of each asset's
    history that the others do not share are discarded, which is the right
    move when assets list or delist at different times and you want the widest
    rectangle they all support. Unlike :func:`align_returns` it never raises on
    ragged input — taking the intersection is the stated intent here — but it
    still rejects an empty intersection, since a panel with no dates is not a
    smaller panel, it is a bug.
    """
    if not returns_by_asset:
        raise ValueError("returns_by_asset is empty")

    columns = {
        asset: _validate_asset_series(series, asset)
        for asset, series in returns_by_asset.items()
    }
    shared = None
    for index in (series.index for series in columns.values()):
        shared = index if shared is None else shared.intersection(index)
    if len(shared) == 0:
        raise ValueError(
            "assets share no common dates; their histories do not overlap"
        )
    return pd.concat(
        {asset: series.loc[shared] for asset, series in columns.items()}, axis=1
    )


def generate_gbm_panel(
    n_assets: int = 5,
    n_periods: int = 1000,
    annual_drift: float = 0.05,
    annual_volatility: float = 0.20,
    periods_per_year: int = 252,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic price panel: ``n_assets`` *independent* GBM columns.

    The assets are mutually independent by construction, which is exactly what
    makes the panel a useful honesty check for cross-sectional strategies: with
    no real cross-sectional structure, a long-short book that ranks these
    assets against each other has nothing to exploit, so any edge it shows is a
    bug. Columns are named ``ASSET_00``, ``ASSET_01``, ... and the index is
    business days from 2020-01-01, matching :func:`generate_gbm_prices`.
    """
    if n_assets < 1:
        raise ValueError(f"n_assets must be >= 1, got {n_assets}")
    if n_periods < 2:
        raise ValueError(f"n_periods must be >= 2, got {n_periods}")
    if annual_volatility < 0:
        raise ValueError(f"annual_volatility must be >= 0, got {annual_volatility}")

    rng = np.random.default_rng(seed)
    dt = 1.0 / periods_per_year
    log_returns = rng.normal(
        loc=(annual_drift - 0.5 * annual_volatility**2) * dt,
        scale=annual_volatility * np.sqrt(dt),
        size=(n_periods - 1, n_assets),
    )
    log_paths = np.vstack([np.zeros(n_assets), np.cumsum(log_returns, axis=0)])
    index = pd.bdate_range("2020-01-01", periods=n_periods)
    columns = [f"ASSET_{i:02d}" for i in range(n_assets)]
    return pd.DataFrame(100.0 * np.exp(log_paths), index=index, columns=columns)
