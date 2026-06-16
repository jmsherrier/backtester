"""Cross-sectional momentum on an independent panel: a null-result demonstration.

Runs the full multi-asset pipeline -- a price panel, a winners-minus-losers
ranking, and the portfolio engine -- on synthetic GBM assets that are
independent by construction. With no real cross-sectional structure, a book
that ranks these assets against each other has nothing to exploit, so the
correct result is a net Sharpe within noise of zero, made strictly worse by
costs.

This is the multi-asset companion to momentum_study.py and the same honesty
check: a long-short book that finds an edge among independent assets is
exposing a bug (most likely lookahead), not alpha.

Run on synthetic data (default) or a folder of real price CSVs, one
``<TICKER>.csv`` (Date,Close) per asset:

    python examples/cross_sectional_study.py
    python examples/cross_sectional_study.py --csv-dir path/to/prices/
"""

from __future__ import annotations

import argparse

from backtester.data import (
    common_window,
    generate_gbm_panel,
    load_price_panel,
    prices_to_returns,
)
from backtester.engine import run_portfolio_backtest
from backtester.execution import BpsCost, ZeroCost
from backtester.signals import cross_sectional_momentum

LOOKBACK = 63  # ~one quarter of daily data
QUANTILE = 0.2  # long the top 20%, short the bottom 20%


def load_returns(args) -> tuple:
    """Return (return matrix, source label, is_synthetic) from the CLI args."""
    if args.csv_dir:
        prices = load_price_panel(args.csv_dir)
        # Per-asset returns first, then the explicit shared-date intersection:
        # real tickers list and delist at different times.
        returns = common_window({t: prices_to_returns(p) for t, p in prices.items()})
        return returns, f"{args.csv_dir} ({returns.shape[1]} assets)", False
    prices = generate_gbm_panel(n_assets=args.assets, n_periods=2520, seed=0)
    returns = prices_to_returns(prices)
    label = f"{args.assets} independent GBM assets (drift 5%, vol 20%, seed 0)"
    return returns, label, True


def format_report(label: str, report: dict[str, float]) -> str:
    return (
        f"{label:<22}"
        f"{report['annualized_return']:>10.2%}"
        f"{report['annualized_volatility']:>10.2%}"
        f"{report['sharpe_ratio']:>9.2f}"
        f"{report['max_drawdown']:>10.2%}"
        f"{report['turnover']:>10.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--csv-dir", help="folder of <TICKER>.csv price files; default: synthetic GBM"
    )
    parser.add_argument("--assets", type=int, default=20)
    parser.add_argument("--lookback", type=int, default=LOOKBACK)
    args = parser.parse_args()

    returns, source, is_synthetic = load_returns(args)
    signal = cross_sectional_momentum(returns, lookback=args.lookback, quantile=QUANTILE)

    gross = run_portfolio_backtest(returns, signal, ZeroCost())
    net = run_portfolio_backtest(returns, signal, BpsCost())

    print(f"data:      {source}")
    print(
        f"periods:   {len(returns)}  |  strategy: {args.lookback}-day "
        f"cross-sectional momentum, top/bottom {QUANTILE:.0%}, dollar-neutral"
    )
    print()
    header = f"{'':<22}{'ann.ret':>10}{'ann.vol':>10}{'sharpe':>9}{'max.dd':>10}{'turnover':>10}"
    print(header)
    print("-" * len(header))
    print(format_report("gross (zero cost)", gross.summary()))
    print(format_report("net (6 bps/trade)", net.summary()))
    print()
    drag = gross.summary()["annualized_return"] - net.summary()["annualized_return"]
    print(f"cost drag: {drag:.2%}/yr at this turnover")
    if is_synthetic:
        print(
            "\nconclusion: ranking independent assets against each other finds no real\n"
            "spread between winners and losers, so a Sharpe indistinguishable from zero --\n"
            "made strictly worse by costs -- is the CORRECT result. A long-short book that\n"
            "shows an edge here is exposing a bug, not alpha."
        )


if __name__ == "__main__":
    main()
