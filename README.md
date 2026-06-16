# backtester

A vectorized, event-aware equity backtesting engine for evaluating systematic trading
strategies **honestly** — with realistic transaction costs, strict train/test separation,
and risk-adjusted performance reporting.

The goal of this project is not to produce an impressive-looking equity curve. It is to
measure whether a signal would actually have made money *after costs* and *out of sample* —
and to be transparent when it would not.

---

## Why this exists

Avoiding: using lookahead information, ignoring transaction
costs, or reporting in-sample performance as if it were predictive. This engine is built to
make those mistakes hard:

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

## Status

In development. Implemented so far:

- **metrics** — annualized return/volatility, Sharpe, max drawdown, hit rate, turnover (unit tested)
- **execution** — transaction-cost models: `ZeroCost` baseline and `BpsCost` (commission + slippage, unit tested)
- **engine** — the core loop: signal → lagged position → gross → net returns. The no-lookahead
  and cost-reconciliation guarantees are executable tests, not just claims. A multi-asset
  variant (`run_portfolio_backtest`) runs a weight matrix against a return matrix — each asset
  lagged and charged on its own notional, summed into one book that nets longs against shorts
- **signals** — time-series momentum (trailing compounded return sign), with a
  truncation-invariance test proving the signal at *t* cannot see past *t*; and
  cross-sectional momentum, which ranks the assets against each other into a
  dollar-neutral, unit-gross winners-minus-losers weight matrix (same no-lookahead proof)
- **data** — strict CSV price loading (reject-don't-repair: no forward-fill, no silent
  dedup), price→return conversion (single asset or panel), and seeded GBM generators for
  runnable examples. Multi-asset return matrices via `align_returns` (rejects ragged panels
  rather than fill or silently inner-join) and `common_window` (the explicit shared-date join);
  `load_price_panel` strict-loads a directory of `<TICKER>.csv` files (alignment left explicit)
- **examples** — `momentum_study.py`: the full pipeline on synthetic random-walk data,
  where the correct answer is *no edge* — a built-in honesty check (run it:
  `python examples/momentum_study.py`, or point it at your own data with `--csv`)
- **validation** — chronological train/test splits (`split_by_fraction`, `split_by_date`,
  accepting a single series or a multi-asset return matrix): every train date precedes every
  test date, the pieces concatenate back to the original exactly, and degenerate splits raise
  instead of returning in-sample data as "out-of-sample".
  Plus `out_of_sample_study`: select a candidate by net Sharpe on the train window, touch
  the test window exactly once, report both numbers so the degradation is the headline.
  And `walk_forward`: refit on each fold (expanding or rolling window) and stitch the
  untouched next blocks into one continuous out-of-sample track — every reported period
  was chosen by a model that had not yet seen it, with no flat reset at fold boundaries.
  Both studies run single- or multi-asset: pass a return series, or a return matrix with
  builders that emit a weight matrix (a cross-sectional book), and the engine is chosen by type
- **examples** — `oos_momentum_study.py`: fits the momentum lookback in-sample on synthetic
  random-walk data and watches the "edge" evaporate out of sample (in-sample Sharpe 0.41 →
  out-of-sample −0.63 on the default seed). `walk_forward_study.py`: refits the lookback every
  quarter — the pick wanders fold to fold and the stitched out-of-sample Sharpe lands at −0.26
  after costs, removing the luck of a single split. Both take real data via `--csv`.
  `cross_sectional_study.py`: the full multi-asset pipeline (panel → ranking → portfolio
  engine) on independent GBM assets, where a winners-minus-losers book has no spread to
  find — another built-in honesty check. `cross_sectional_oos_study.py`: the same book run
  through both validation studies — a single 70/30 split and a walk-forward — showing the
  in-sample lookback pick degrade out of sample to a Sharpe with |t| < 2 (one panel's noise).
  The two cross-sectional examples take real data via `--csv-dir` (a folder of `<TICKER>.csv`)

Up next: a mean-reversion signal and a documented real-data case study.

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

MIT
