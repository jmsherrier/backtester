"""Momentum on a random walk: a null-result demonstration.

Runs the time-series momentum strategy over synthetic GBM prices and reports
gross vs net performance. GBM returns are independent by construction, so
there is no trend to follow — the *correct* result is a Sharpe within noise
of zero gross, made strictly worse by costs.

That is the point of the example. A backtester should be trusted on its
negative results before its positive ones: if this study ever reports a
significant edge, the framework has a bug (almost certainly lookahead).
The companion test in tests/test_signals.py guards the same invariant.

Run against synthetic data (default) or your own price CSV:

    python examples/momentum_study.py
    python examples/momentum_study.py --csv path/to/prices.csv
"""

from __future__ import annotations

import argparse

from backtester.data import generate_gbm_prices, load_prices_csv, prices_to_returns
from backtester.engine import run_backtest
from backtester.execution import BpsCost, ZeroCost
from backtester.signals import time_series_momentum

LOOKBACK = 63  # ~one quarter of daily data


def format_report(label: str, report: dict[str, float]) -> str:
    return (
        f"{label:<22}"
        f"{report['annualized_return']:>10.2%}"
        f"{report['annualized_volatility']:>10.2%}"
        f"{report['sharpe_ratio']:>9.2f}"
        f"{report['max_drawdown']:>10.2%}"
        f"{report['hit_rate']:>9.2%}"
        f"{report['turnover']:>10.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", help="price CSV (Date,Close); default: synthetic GBM")
    parser.add_argument("--lookback", type=int, default=LOOKBACK)
    args = parser.parse_args()

    if args.csv:
        prices = load_prices_csv(args.csv)
        source = args.csv
    else:
        prices = generate_gbm_prices(n_periods=2520, seed=0)  # ~10 years daily
        source = "synthetic GBM (drift 5%, vol 20%, seed 0) -- no real structure"

    returns = prices_to_returns(prices)
    signal = time_series_momentum(returns, lookback=args.lookback)

    gross = run_backtest(returns, signal, ZeroCost())
    net = run_backtest(returns, signal, BpsCost())  # 1 bp commission + 5 bps slippage

    print(f"data:      {source}")
    print(f"periods:   {len(returns)}  |  strategy: {args.lookback}-day ts-momentum")
    print()
    header = f"{'':<22}{'ann.ret':>10}{'ann.vol':>10}{'sharpe':>9}{'max.dd':>10}{'hit':>9}{'turnover':>10}"
    print(header)
    print("-" * len(header))
    print(format_report("gross (zero cost)", gross.summary()))
    print(format_report("net (6 bps/trade)", net.summary()))
    print()
    drag = gross.summary()["annualized_return"] - net.summary()["annualized_return"]
    print(f"cost drag: {drag:.2%}/yr at this turnover")
    if not args.csv:
        print(
            "\nconclusion: on a random walk momentum has no edge to find, so a Sharpe\n"
            "indistinguishable from zero -- made strictly worse by costs -- is the\n"
            "CORRECT result. Treat any framework that finds a real edge here as broken."
        )


if __name__ == "__main__":
    main()
