# backtester

A vectorized, event-aware equity backtesting engine for evaluating systematic trading
strategies **honestly** — with realistic transaction costs, strict train/test separation,
and risk-adjusted performance reporting.

The goal of this project is not to produce an impressive-looking equity curve. It is to
measure whether a signal would actually have made money *after costs* and *out of sample* —
and to be transparent when it would not.

---

## Why this exists

Most student backtests quietly cheat: they use lookahead information, ignore transaction
costs, or report in-sample performance as if it were predictive. This engine is built to
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
  data/         # price/return loading + universe construction
  signals/      # strategy signal generators (momentum, mean-reversion, ...)
  execution/    # transaction-cost and slippage models
  engine/       # the core backtest loop: signals -> positions -> P&L
  metrics/      # Sharpe, drawdown, turnover, hit rate, etc.
tests/          # unit tests (lookahead checks, cost accounting, metric math)
examples/       # runnable strategy studies with written conclusions
```

## Status

In development. Implemented so far:

- **metrics** — annualized return/volatility, Sharpe, max drawdown, hit rate, turnover (unit tested)
- **execution** — transaction-cost models: `ZeroCost` baseline and `BpsCost` (commission + slippage, unit tested)

Up next: the core engine loop (signals → positions → net P&L).

## Getting started

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
pytest                         # run the test suite
```

## Stack

Python 3.12 · NumPy · Pandas · pytest

## License

MIT
