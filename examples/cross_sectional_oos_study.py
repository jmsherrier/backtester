"""Out-of-sample + walk-forward cross-sectional study, driven by the portfolio engine.

This is the multi-asset analogue of oos_momentum_study.py and walk_forward_study.py:
the same select-then-evaluate discipline, but the candidate is a cross-sectional
lookback, the signal is a dollar-neutral weight matrix, and every backtest runs
through the multi-asset engine. Two views of the same question:

1. A single 70/30 split: pick the lookback by net Sharpe in-sample, then look at
   the held-out window exactly once.
2. Walk-forward: refit the lookback every quarter on an expanding window and
   stitch the untouched next-quarter blocks into one continuous track.

On the default independent-GBM panel there is no real spread between winners and
losers, so a lookback that looks good in-sample should not survive out of sample.
A single random panel can still realize a smallish non-zero out-of-sample Sharpe
by chance — what the study shows is that it is not statistically distinguishable
from zero (the printed t-statistic is well under 2) and that the in-sample edge
degrades. A *significant* edge here would mean the framework is leaking.

    python examples/cross_sectional_oos_study.py
"""

from __future__ import annotations

import argparse

from backtester.data import generate_gbm_panel, prices_to_returns
from backtester.signals import cross_sectional_momentum
from backtester.validation import out_of_sample_study, walk_forward

LOOKBACKS = (21, 63, 126, 252)  # ~one month to one year of daily data
QUANTILE = 0.2  # long the top 20%, short the bottom 20%
TRAIN_SIZE = 504  # ~2 years to fit on
TEST_SIZE = 63  # ~1 quarter evaluated, then refit


def candidates() -> dict:
    return {
        lookback: (
            lambda r, lb=lookback: cross_sectional_momentum(r, lookback=lb, quantile=QUANTILE)
        )
        for lookback in LOOKBACKS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--assets", type=int, default=20)
    args = parser.parse_args()

    prices = generate_gbm_panel(n_assets=args.assets, n_periods=2520, seed=0)
    returns = prices_to_returns(prices)
    source = f"{args.assets} independent GBM assets (seed 0) -- no cross-sectional structure"

    print(f"data:      {source}")
    print(
        f"periods:   {len(returns)}  |  candidates: {LOOKBACKS} day lookbacks, "
        f"top/bottom {QUANTILE:.0%}  |  costs: 6 bps/trade"
    )

    # --- 1. Single 70/30 split -------------------------------------------------
    oos = out_of_sample_study(returns, candidates(), train_fraction=0.7)
    print("\n[1] single split (select in-sample, evaluate held-out window once)")
    print("  in-sample net Sharpe by lookback (selection happens here):")
    for lookback in LOOKBACKS:
        marker = "  <- selected" if lookback == oos.selected else ""
        print(f"    {lookback:>4}d  {oos.train_scores[lookback]:>6.2f}{marker}")
    print(
        f"  in-sample Sharpe {oos.in_sample['sharpe_ratio']:>6.2f}  ->  "
        f"out-of-sample Sharpe {oos.out_of_sample['sharpe_ratio']:>6.2f}  "
        f"(degradation {oos.in_sample['sharpe_ratio'] - oos.out_of_sample['sharpe_ratio']:.2f})"
    )

    # --- 2. Walk-forward -------------------------------------------------------
    wf = walk_forward(returns, candidates(), train_size=TRAIN_SIZE, test_size=TEST_SIZE)
    picks = [fold.selected for fold in wf.folds]
    report = wf.summary
    n_years = report["n_periods"] / 252
    # A Sharpe's t-statistic over the sample is roughly Sharpe * sqrt(years);
    # |t| < 2 means the track is not distinguishable from a zero-edge book.
    t_stat = report["sharpe_ratio"] * (n_years**0.5)
    print(f"\n[2] walk-forward ({len(wf.folds)} quarterly refits on an expanding window)")
    print(f"  lookback picked per fold: {picks}")
    print(
        f"  stitched out-of-sample track: {int(report['n_periods'])} periods, "
        f"Sharpe (after cost) {report['sharpe_ratio']:>6.2f} (t = {t_stat:.2f}), "
        f"max dd {report['max_drawdown']:.2%}, turnover {report['turnover']:.1f}"
    )
    print(
        "\nconclusion: there is no real spread between winners and losers on an\n"
        "independent panel, so the in-sample pick degrades out of sample and the\n"
        "walk-forward track's Sharpe is not statistically distinguishable from zero\n"
        "(|t| well under 2). That is the CORRECT result -- a *significant* multi-asset\n"
        "edge here would be exposing a bug, not alpha."
    )


if __name__ == "__main__":
    main()
