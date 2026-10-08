"""Momentum vs mean-reversion: which wins after costs, out of sample?

A head-to-head case study. Both signal families compete as candidates in the
same out-of-sample study: the lookback (and reversion threshold) that scores
best by net Sharpe on the train window is chosen, then judged once on the
held-out window. A walk-forward pass then shows which family the refit keeps
picking as the data rolls on.

The two strategies are natural opposites — momentum rides a trend, mean-reversion
fades it — so on any given asset at most one should look good, and the honest
question is whether that in-sample edge survives out of sample after costs.

On the default synthetic GBM data the answer is neither: a random walk has no
trend to ride and no level to revert to, so both degrade to noise. Point it at a
real price series to run the actual contest:

    python examples/momentum_vs_reversion.py
    python examples/momentum_vs_reversion.py --csv path/to/prices.csv
"""

from __future__ import annotations

import argparse

from backtester.data import generate_gbm_prices, load_prices_csv, prices_to_returns
from backtester.signals import mean_reversion, time_series_momentum
from backtester.validation import out_of_sample_study, walk_forward

# Momentum lookbacks and mean-reversion (lookback, entry_z) configs to pit
# against each other. Labels carry the family so the winner is self-describing.
CANDIDATES = {
    "mom-21": lambda r: time_series_momentum(r, lookback=21),
    "mom-63": lambda r: time_series_momentum(r, lookback=63),
    "mom-252": lambda r: time_series_momentum(r, lookback=252),
    "rev-21": lambda r: mean_reversion(r, lookback=21, entry_z=1.0),
    "rev-63": lambda r: mean_reversion(r, lookback=63, entry_z=1.0),
}
TRAIN_SIZE = 504  # ~2 years to fit on
TEST_SIZE = 63  # ~1 quarter evaluated, then refit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", help="price CSV (Date,Close); default: synthetic GBM")
    args = parser.parse_args()

    if args.csv:
        prices = load_prices_csv(args.csv)
        source = args.csv
    else:
        prices = generate_gbm_prices(n_periods=2520, seed=0)
        source = "synthetic GBM (seed 0) -- no trend, no level to revert to"
    returns = prices_to_returns(prices)

    print(f"data:      {source}")
    print(f"periods:   {len(returns)}  |  costs: 6 bps/trade")

    # --- Single 70/30 split: select in-sample, judge the held-out window once ---
    oos = out_of_sample_study(returns, CANDIDATES, train_fraction=0.7)
    print("\n[1] single split -- in-sample net Sharpe (selection happens here):")
    for label, score in sorted(oos.train_scores.items(), key=lambda kv: -kv[1]):
        marker = "  <- selected" if label == oos.selected else ""
        print(f"    {label:<8}{score:>6.2f}{marker}")
    print(
        f"  winner {oos.selected}: in-sample Sharpe {oos.in_sample['sharpe_ratio']:.2f}"
        f"  ->  out-of-sample {oos.out_of_sample['sharpe_ratio']:.2f}"
        f"  (degradation {oos.in_sample['sharpe_ratio'] - oos.out_of_sample['sharpe_ratio']:.2f})"
    )

    # --- Walk-forward: which family does the refit keep choosing? ---
    wf = walk_forward(returns, CANDIDATES, train_size=TRAIN_SIZE, test_size=TEST_SIZE)
    picks = [fold.selected for fold in wf.folds]
    n_mom = sum(p.startswith("mom") for p in picks)
    report = wf.summary
    t_stat = report["sharpe_ratio"] * (report["n_periods"] / 252) ** 0.5
    print(f"\n[2] walk-forward ({len(wf.folds)} quarterly refits):")
    print(f"  family picked: momentum {n_mom}/{len(picks)} folds, reversion {len(picks) - n_mom}")
    print(
        f"  stitched out-of-sample Sharpe {report['sharpe_ratio']:.2f} (t = {t_stat:.2f}), "
        f"turnover {report['turnover']:.1f}"
    )

    if not args.csv:
        print(
            "\nconclusion: on a random walk there is nothing for either family to find,\n"
            "so the in-sample winner degrades out of sample and the walk-forward track is\n"
            "not distinguishable from zero (|t| < 2). That is the CORRECT result -- the\n"
            "contest only means something on real prices, via --csv."
        )


if __name__ == "__main__":
    main()
