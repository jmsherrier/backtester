"""Walk-forward momentum study: many refits, one continuous out-of-sample track.

The single-split study (`oos_momentum_study.py`) reports one out-of-sample
number, which depends on where the split happened to land. This walk-forward
version refits the momentum lookback every quarter on an expanding window and
strings the untouched next-quarter blocks into one continuous out-of-sample
return series — every period it scores was chosen by a model that had not yet
seen it.

On the default synthetic GBM data the verdict only sharpens: the selected
lookback wanders from fold to fold (it is chasing noise, so there is nothing
stable to lock onto), and the stitched out-of-sample Sharpe sits near or below
zero after costs. A walk-forward edge on a random walk would mean the framework
is leaking.

Run against synthetic data (default) or your own price CSV:

    python examples/walk_forward_study.py
    python examples/walk_forward_study.py --csv path/to/prices.csv
"""

from __future__ import annotations

import argparse

from backtester.data import generate_gbm_prices, load_prices_csv, prices_to_returns
from backtester.signals import time_series_momentum
from backtester.validation import walk_forward

LOOKBACKS = (21, 63, 126, 252)  # ~one month to one year of daily data
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
        prices = generate_gbm_prices(n_periods=2520, seed=0)  # ~10 years daily
        source = "synthetic GBM (drift 5%, vol 20%, seed 0) -- no real structure"

    returns = prices_to_returns(prices)
    candidates = {
        lookback: (lambda r, lb=lookback: time_series_momentum(r, lb))
        for lookback in LOOKBACKS
    }
    result = walk_forward(
        returns, candidates, train_size=TRAIN_SIZE, test_size=TEST_SIZE
    )

    print(f"data:      {source}")
    print(
        f"periods:   {len(returns)}  |  "
        f"refit every {TEST_SIZE} on an expanding {TRAIN_SIZE}+ window  |  "
        f"costs: 6 bps/trade"
    )
    print()
    print("per-fold selection and its block's out-of-sample Sharpe:")
    print(f"  {'fold window':<26}{'train':>7}{'pick':>7}{'oos.sharpe':>12}")
    print("  " + "-" * 50)
    for i, fold in enumerate(result.folds):
        window = f"{fold.test_start.date()}..{fold.test_end.date()}"
        print(
            f"  {window:<26}{fold.n_train:>7}{fold.selected:>7}"
            f"{fold.oos_sharpe:>12.2f}"
        )
    print()
    report = result.summary
    print(f"stitched out-of-sample track: {int(report['n_periods'])} periods")
    print(f"  annualized return:   {report['annualized_return']:>8.2%}")
    print(f"  annualized vol:      {report['annualized_volatility']:>8.2%}")
    print(f"  Sharpe (after cost): {report['sharpe_ratio']:>8.2f}")
    print(f"  max drawdown:        {report['max_drawdown']:>8.2%}")
    print(f"  turnover:            {report['turnover']:>8.1f}")
    if not args.csv:
        print(
            "\nconclusion: the chosen lookback drifts fold to fold because there is no\n"
            "stable trend to find on a random walk, and the stitched out-of-sample\n"
            "Sharpe lands near or below zero after costs. That is the CORRECT result --\n"
            "walk-forward removes the luck of a single split, and the edge still isn't\n"
            "there because it never was."
        )


if __name__ == "__main__":
    main()
