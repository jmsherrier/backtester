"""Walk-forward evaluation: many refits, one continuous out-of-sample track.

A single train/test split (see :mod:`backtester.validation.study`) reports one
out-of-sample number — honest, but a sample of one, and sensitive to where the
split happened to land. Walk-forward generalizes it: step through the data in
folds, reselecting the best candidate on each fold's train window and recording
its performance on the *next* untouched block, then stitch those out-of-sample
blocks into one continuous return series. Every reported period was chosen by a
model that had not yet seen it.

Two regimes for the train window:

- *anchored* (default): the train window grows from a fixed start, so each
  refit sees all history to date — the natural choice when more data is better
  and the relationship is assumed stable.
- *rolling*: the train window is a fixed length that slides forward, so old
  data ages out — appropriate when you believe the world drifts and stale
  history misleads.

Fold boundaries carry no artifact: each fold's positions and costs are computed
on the full history through that fold, then sliced to the test block, so the
position carried in from the last train day is the one actually held on the
first test day — not an artificial flat reset. Leakage of the future into the
past is impossible by the signal's truncation-invariance contract (enforced in
the signals tests), the same guarantee the single-split study relies on.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.engine import run_backtest
from backtester.execution.costs import BpsCost, CostModel
from backtester.metrics.performance import sharpe_ratio, summary
from backtester.validation.study import SignalBuilder


@dataclass(frozen=True)
class Fold:
    """One refit step: what was selected on train and how it did on test.

    ``test_start``/``test_end`` are the inclusive date bounds of the block this
    fold contributed to the stitched out-of-sample series; ``oos_sharpe`` is
    that block's net Sharpe in isolation, useful for seeing whether the
    out-of-sample track is steady or driven by one lucky fold.
    """

    selected: Hashable
    n_train: int
    n_test: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    oos_sharpe: float


@dataclass(frozen=True)
class WalkForwardResult:
    """The stitched out-of-sample track plus the per-fold selection history.

    ``oos_returns`` and ``oos_positions`` are the concatenated out-of-sample
    blocks across every fold — a single continuous series, never overlapping.
    ``summary`` is the standard metric set on that series; its
    ``sharpe_ratio`` is the headline walk-forward number. ``folds`` records
    what was selected at each step, so a drifting or unstable selection is
    visible rather than averaged away.
    """

    oos_returns: pd.Series
    oos_positions: pd.Series
    summary: dict[str, float]
    folds: list[Fold]


def walk_forward(
    returns: pd.Series,
    candidates: Mapping[Hashable, SignalBuilder],
    train_size: int,
    test_size: int,
    cost_model: CostModel | None = None,
    anchored: bool = True,
) -> WalkForwardResult:
    """Roll a refit-then-evaluate loop across ``returns``, stitching the
    out-of-sample blocks into one continuous track.

    Parameters
    ----------
    returns : pd.Series
        Periodic simple returns of the traded asset, sorted ascending.
    candidates : Mapping[Hashable, SignalBuilder]
        Label -> signal builder, as in
        :func:`backtester.validation.study.out_of_sample_study`.
    train_size : int
        Number of periods in the first fold's train window. Under
        ``anchored`` the window grows from the start by ``test_size`` each
        fold; under rolling it stays this length and slides.
    test_size : int
        Number of periods evaluated out-of-sample per fold, and the amount the
        window advances between folds. The final fold may be shorter if the
        data does not divide evenly — a short tail is still out-of-sample, so
        it is reported rather than discarded.
    cost_model : CostModel, optional
        Defaults to :class:`BpsCost`; a walk-forward result is a reported
        number, so costs are realistic by default. Selection and evaluation
        use the same model.
    anchored : bool
        ``True`` (default) for an expanding train window, ``False`` for a
        fixed-length rolling one.

    Returns
    -------
    WalkForwardResult

    Raises
    ------
    ValueError
        If ``candidates`` is empty; if ``train_size``/``test_size`` are not
        positive; if there are not enough periods for even one fold; or if
        some fold has no candidate with a finite in-sample Sharpe.
    """
    if not candidates:
        raise ValueError("candidates is empty")
    if train_size < 1 or test_size < 1:
        raise ValueError(
            f"train_size and test_size must be positive, got "
            f"train_size={train_size}, test_size={test_size}"
        )
    if not isinstance(returns, pd.Series):
        raise TypeError(f"returns must be a pandas Series, got {type(returns).__name__}")
    if not returns.index.is_monotonic_increasing:
        raise ValueError("returns index must be sorted ascending")
    n = len(returns)
    if n < train_size + test_size:
        raise ValueError(
            f"need at least train_size + test_size = {train_size + test_size} "
            f"periods for one fold, got {n}"
        )
    if cost_model is None:
        cost_model = BpsCost()

    oos_returns_blocks: list[pd.Series] = []
    oos_positions_blocks: list[pd.Series] = []
    folds: list[Fold] = []

    train_end = train_size
    while train_end < n:
        train_start = train_end - train_size if not anchored else 0
        test_end = min(train_end + test_size, n)
        train = returns.iloc[train_start:train_end]
        test_index = returns.index[train_end:test_end]

        selected = _select(train, candidates, cost_model)

        # Compute positions/costs on the full history through this fold, then
        # slice to the test block: the position held on the first test day is
        # the one carried in from the last train day, not an artificial reset.
        signal_through_fold = candidates[selected](returns.iloc[:test_end])
        result = run_backtest(
            returns.iloc[:test_end], signal_through_fold, cost_model
        )
        block_returns = result.net_returns.loc[test_index]
        block_positions = result.positions.loc[test_index]

        oos_returns_blocks.append(block_returns)
        oos_positions_blocks.append(block_positions)
        folds.append(
            Fold(
                selected=selected,
                n_train=len(train),
                n_test=len(test_index),
                test_start=test_index[0],
                test_end=test_index[-1],
                # From the same continuous block, not a re-run on the slice:
                # a fold's number should match its contribution to the track.
                oos_sharpe=sharpe_ratio(block_returns),
            )
        )
        train_end += test_size

    oos_returns = pd.concat(oos_returns_blocks)
    oos_positions = pd.concat(oos_positions_blocks)
    return WalkForwardResult(
        oos_returns=oos_returns,
        oos_positions=oos_positions,
        summary=_summarize(oos_returns, oos_positions),
        folds=folds,
    )


def _select(
    train: pd.Series,
    candidates: Mapping[Hashable, SignalBuilder],
    cost_model: CostModel,
) -> Hashable:
    """Pick the candidate with the best finite net Sharpe on the train window."""
    scores = {
        label: run_backtest(train, build(train), cost_model).summary()["sharpe_ratio"]
        for label, build in candidates.items()
    }
    finite = {label: s for label, s in scores.items() if np.isfinite(s)}
    if not finite:
        raise ValueError(
            "no candidate produced a finite in-sample Sharpe on a fold; "
            "nothing to select"
        )
    return max(finite, key=finite.__getitem__)


def _summarize(oos_returns: pd.Series, oos_positions: pd.Series) -> dict[str, float]:
    """Standard metric set on the stitched out-of-sample series."""
    return summary(oos_returns, positions=oos_positions)
