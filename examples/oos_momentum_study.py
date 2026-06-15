"""Out-of-sample momentum study: fit the lookback, then face the test window.

Picks the best momentum lookback by net Sharpe on the first 70% of the data,
then evaluates that single choice — once — on the held-out 30%. Both numbers
are reported side by side so the in-sample-to-out-of-sample degradation is
the headline, not a footnote.

On the default synthetic GBM data the correct outcome is sobering by design:
the "best" in-sample lookback is the one that happened to fit that window's
noise, and out of sample the edge should evaporate. If the out-of-sample
Sharpe here were ever reliably positive, the framework would be leaking.

Run against synthetic data (default) or your own price CSV:

    python examples/oos_momentum_study.py
    python examples/oos_momentum_study.py --csv path/to/prices.csv
"""

from __future__ import annotations

import argparse

from backtester.data import generate_gbm_prices, load_prices_csv, prices_to_returns
from backtester.signals import time_series_momentum
from backtester.validation import out_of_sample_study

LOOKBACKS = (21, 63, 126, 252)  # ~one month to one year of daily data
TRAIN_FRACTION = 0.7


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
    parser.add_argument("--csv", help="price CSV (Date,Close); default: synthetic GBM")
    args = parser.parse_args()

    if args.csv:
        prices = load_prices_csv(args.csv)
        source = args.csv
    else:
        prices = generate_gbm_prices(n_periods=2520, seed=0)  # ~10 years daily
        source = "synthetic GBM (drift 5%, vol 20%, seed 0) -- no real structure"

    returns = prices_to_returns(prices)
    candidates = {
        lookback: (lambda r, lb=lookback: time_series_momentum(r, lb))
        for lookback in LOOKBACKS
    }
    result = out_of_sample_study(returns, candidates, train_fraction=TRAIN_FRACTION)

    print(f"data:      {source}")
    print(
        f"periods:   {len(returns)}  |  "
        f"train: {int(result.in_sample['n_periods'])}  "
        f"test: {int(result.out_of_sample['n_periods'])}  |  costs: 6 bps/trade"
    )
    print()
    print("in-sample net Sharpe by lookback (selection happens here):")
    for lookback in LOOKBACKS:
        marker = "  <- selected" if lookback == result.selected else ""
        print(f"  {lookback:>4}d  {result.train_scores[lookback]:>6.2f}{marker}")
    print()
    header = f"{'':<22}{'ann.ret':>10}{'ann.vol':>10}{'sharpe':>9}{'max.dd':>10}{'turnover':>10}"
    print(header)
    print("-" * len(header))
    print(format_report("in-sample (fit)", result.in_sample))
    print(format_report("out-of-sample", result.out_of_sample))
    print()
    degradation = (
        result.in_sample["sharpe_ratio"] - result.out_of_sample["sharpe_ratio"]
    )
    print(f"Sharpe degradation out of sample: {degradation:.2f}")
    if not args.csv:
        print(
            "\nconclusion: the selected lookback won the train window by fitting its\n"
            "noise -- there is nothing real to fit on a random walk -- so the edge\n"
            "should not survive out of sample. A near-zero or negative out-of-sample\n"
            "Sharpe is the CORRECT result here, and the degradation line above is\n"
            "exactly the gap that in-sample-only backtests hide."
        )


if __name__ == "__main__":
    main()
