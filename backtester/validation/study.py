"""Out-of-sample parameter studies.

The procedure this module pins down:

1. Split the data chronologically.
2. Score every candidate on the train window only and pick the best.
3. Evaluate the winner on the test window exactly once, and report both
   numbers so in-sample-to-out-of-sample degradation is visible.

One boundary subtlety is handled deliberately: when evaluating the test
window, the winner's signal is built from the *full* history, not the test
slice alone. At any test date the train window is past information — a real
trader would have had it — so denying the signal its warm-up history would
understate performance rather than protect against leakage. Leakage in the
other direction (future into past) is impossible by the signal's own
truncation-invariance contract, which is enforced by tests, not assumed.

The same procedure runs single- or multi-asset: pass a return *series* with
builders that emit a single-asset signal, or a return *matrix* with builders
that emit a weight matrix (e.g. a cross-sectional ranker). The engine is
selected from the input type, so the selection-and-degradation story is told
identically for one asset or a long-short book.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.engine import run_backtest, run_portfolio_backtest
from backtester.execution.costs import BpsCost, CostModel
from backtester.validation.split import Data, split_by_fraction

# Builds a signal from the return history available at the time: takes returns
# through date t, produces an aligned signal (a single-asset position series,
# or a weight matrix for a panel) using only information through each date.
SignalBuilder = Callable[[Data], Data]


def run_engine(returns: Data, signal: Data, cost_model: CostModel):
    """Dispatch to the single- or multi-asset engine by input type.

    A ``DataFrame`` of returns is a panel and runs through the portfolio
    engine; a ``Series`` is a single asset. Both results expose ``.summary()``
    and a ``net_returns`` Series, so callers stay engine-agnostic.
    """
    if isinstance(returns, pd.DataFrame):
        return run_portfolio_backtest(returns, signal, cost_model)
    return run_backtest(returns, signal, cost_model)


@dataclass(frozen=True)
class StudyResult:
    """Everything an out-of-sample study reports.

    ``train_scores`` holds every candidate's in-sample net Sharpe — the full
    selection picture, not just the winner — so a reader can see whether the
    chosen parameter won decisively or by noise. ``in_sample`` and
    ``out_of_sample`` are the standard metric summaries of the winner on the
    two windows; the honest headline is ``out_of_sample["sharpe_ratio"]``.
    """

    selected: Hashable
    train_scores: dict[Hashable, float]
    in_sample: dict[str, float]
    out_of_sample: dict[str, float]


def out_of_sample_study(
    returns: Data,
    candidates: Mapping[Hashable, SignalBuilder],
    cost_model: CostModel | None = None,
    train_fraction: float = 0.7,
) -> StudyResult:
    """Select a strategy on the train window, evaluate it once on the test window.

    Parameters
    ----------
    returns : pd.Series | pd.DataFrame
        Periodic simple returns, sorted ascending — a single asset (Series) or
        an aligned return matrix (DataFrame); the matching engine is chosen
        automatically.
    candidates : Mapping[Hashable, SignalBuilder]
        Label -> signal builder. Each builder receives a return history and
        must return an aligned signal using only information through each
        date (the no-lookahead contract every signal in this package tests).
        For a panel the signal is a weight matrix shaped like ``returns``.
    cost_model : CostModel, optional
        Defaults to :class:`BpsCost` — unlike the raw engine, a study's
        output is a *reported* number, so costs default to realistic here.
        Selection and evaluation use the same model.
    train_fraction : float
        Passed to :func:`split_by_fraction`.

    Returns
    -------
    StudyResult

    Raises
    ------
    ValueError
        If ``candidates`` is empty, or no candidate achieves a finite
        in-sample Sharpe (e.g. every candidate stayed flat) — selecting
        among unusable scores would just be picking dict order.
    """
    if not candidates:
        raise ValueError("candidates is empty")
    if cost_model is None:
        cost_model = BpsCost()

    split = split_by_fraction(returns, train_fraction)

    train_scores = {
        label: run_engine(split.train, build(split.train), cost_model)
        .summary()["sharpe_ratio"]
        for label, build in candidates.items()
    }

    finite = {label: s for label, s in train_scores.items() if np.isfinite(s)}
    if not finite:
        raise ValueError(
            "no candidate produced a finite in-sample Sharpe; nothing to select"
        )
    selected = max(finite, key=finite.__getitem__)
    build = candidates[selected]

    in_sample = run_engine(split.train, build(split.train), cost_model).summary()

    # The single touch of the test window. The signal is built on the full
    # history (see module docstring) and only its test-window values are
    # evaluated; the engine still starts the window flat, so the first test
    # period is conservatively uninvested.
    full_signal = build(returns)
    out_of_sample = run_engine(
        split.test, full_signal.loc[split.test.index], cost_model
    ).summary()

    return StudyResult(
        selected=selected,
        train_scores=train_scores,
        in_sample=in_sample,
        out_of_sample=out_of_sample,
    )
