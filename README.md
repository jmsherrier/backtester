# backtester

A vectorized, event-aware equity backtesting engine for evaluating systematic trading
strategies **honestly** — with realistic transaction costs, strict train/test separation,
and risk-adjusted performance reporting.

The goal of this project is not to produce an impressive-looking equity curve. It is to
measure whether a signal would actually have made money *after costs* and *out of sample* —
and to be transparent when it would not.

---

## Why this exists

Most backtests flatter a strategy in three ways: they use information that wasn't available
yet, they ignore transaction costs, and they report in-sample performance as if it were
predictive. This engine is built to make those mistakes hard:

- **No lookahead by construction** — signals at time *t* may only use data available at *t*.
- **Costs are not optional** — every fill pays commission and slippage.
- **In-sample and out-of-sample are separated up front** — parameters are fit on one window
  and evaluated on another that was never touched during fitting.

The headline metric is **out-of-sample Sharpe after costs**, reported alongside the in-sample
number so the degradation is visible rather than hidden.

---

## Project layout

```
backtester/
  data/         # price/return loading + multi-asset return matrices
  signals/      # strategy signal generators (time-series + cross-sectional)
  execution/    # transaction-cost and slippage models
  engine/       # the core backtest loop: signals -> positions -> P&L (single- and multi-asset)
  metrics/      # Sharpe, drawdown, turnover, hit rate, etc.
  validation/   # train/test splits, out-of-sample + walk-forward studies
tests/          # unit tests (lookahead checks, cost accounting, metric math)
examples/       # runnable strategy studies with written conclusions
```

## Results so far

The included studies run on synthetic random-walk data, where the correct answer is *no edge*,
so they double as honesty checks. Each takes real prices via `--csv` (or `--csv-dir` for a
folder of `<TICKER>.csv` files).

| Study | What it shows |
| --- | --- |
| `oos_momentum_study.py` | Momentum lookback fit in-sample: Sharpe **0.41 in-sample → −0.63 out of sample** |
| `walk_forward_study.py` | Quarterly refits; the chosen lookback wanders and stitched OOS Sharpe lands at **−0.26** after costs |
| `cross_sectional_oos_study.py` | Winners-minus-losers book degrades out of sample to \|t\| < 2, i.e. noise |
| `momentum_vs_reversion.py` | Momentum and mean-reversion compete in one study; the in-sample winner is judged after costs out of sample |

## What's implemented

- **engine**: signal → lagged position → gross → net returns, single-asset and multi-asset
  (`run_portfolio_backtest`, each asset charged on its own notional). The no-lookahead and
  cost-reconciliation guarantees are executable tests.
- **signals**: time-series momentum, cross-sectional momentum (dollar-neutral, unit-gross), and
  Bollinger-style mean reversion, each with a truncation-invariance test proving the signal at
  *t* cannot see past *t*.
- **execution**: `ZeroCost` baseline and `BpsCost` (commission + slippage).
- **metrics**: annualized return and volatility, Sharpe, max drawdown, hit rate, turnover.
- **data**: strict CSV loading that rejects bad data instead of repairing it (no forward-fill,
  no silent dedup or inner-join), price panels, and seeded GBM generators.
- **validation**: chronological train/test splits, a one-shot out-of-sample study, and
  walk-forward (expanding or rolling) that stitches untouched folds into one OOS track.

Up next: volatility-targeted position sizing (signals currently size at ±1).

## Getting started

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e .[dev]          # editable install + test deps
pytest                         # run the test suite
```

## Stack

Python 3.12 · NumPy · Pandas · pytest

## License

[MIT](LICENSE)
