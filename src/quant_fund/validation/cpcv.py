"""Combinatorial purged cross-validation (López de Prado).

Path count follows López de Prado, *Advances in Financial Machine Learning*
(2018), chapter 12: with ``N`` groups and ``k`` test groups there are
``C(N, k)`` splits and ``φ = C(N-1, k-1)`` backtest paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from math import comb
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from quant_fund.validation.purging import purge_mask
from quant_fund.validation.walk_forward import Fold, session_index

IntArray = NDArray[np.int64]

#: Conservative selection criteria for :func:`rank_configs`. ``"mean"`` is the
#: average reconstructed path — the in-sample-best-style criterion an
#: overfit config wins on. ``"worst_path"`` is the minimum over paths, so a
#: config that only works on one lucky split is ranked by its worst split.
#: Nothing in the repo selects on these by default; callers opt in.
RankCriterion = Literal["mean", "worst_path"]
RANK_CRITERIA: tuple[str, ...] = ("mean", "worst_path")


def combinatorial_purged_cv(
    times: list[datetime],
    n_groups: int,
    n_test_groups: int,
    horizon_bars: int,
    embargo_bars: int,
) -> list[Fold]:
    """Split the timeline into ``n_groups`` sequential groups; every combination
    of ``n_test_groups`` is the test set, train is the purged/embargoed complement.

    Purge and embargo are applied **per contiguous test group**. Using the
    min/max span of a non-contiguous test combination would incorrectly wipe
    intervening train groups.

    Raises
    ------
    ValueError
        Invalid group counts, negative horizon/embargo, or fewer unique dates
        than ``n_groups`` (empty groups would break fold-count integrity).
    """
    if n_groups < 2 or n_test_groups < 1 or n_test_groups >= n_groups:
        raise ValueError("invalid CPCV group counts")
    if horizon_bars < 0:
        raise ValueError("horizon_bars must be non-negative")
    if embargo_bars < 0:
        raise ValueError("embargo_bars must be non-negative")
    uniq = sorted(set(times))
    n = len(uniq)
    if n == 0:
        return []
    if n < n_groups:
        raise ValueError(f"need at least n_groups={n_groups} unique dates for CPCV, got {n}")
    bounds = [int(i * n / n_groups) for i in range(n_groups + 1)]
    groups = [uniq[bounds[i] : bounds[i + 1]] for i in range(n_groups)]
    if any(len(g) == 0 for g in groups):
        raise ValueError("CPCV group bounds produced an empty group")
    idx = session_index(uniq)
    folds: list[Fold] = []
    for test_ids in combinations(range(n_groups), n_test_groups):
        test_times: list[datetime] = []
        for g in test_ids:
            test_times.extend(groups[g])
        train_times: list[datetime] = []
        for g in range(n_groups):
            if g not in test_ids:
                train_times.extend(groups[g])
        if not test_times or not train_times:
            continue
        purged = list(train_times)
        for g in test_ids:
            block = groups[g]
            if not block:
                continue
            keep = purge_mask(purged, block[0], block[-1], horizon_bars, session_index=idx)
            purged = [t for t, k in zip(purged, keep, strict=True) if k]
            lo, hi = idx[block[0]], idx[block[-1]]
            purged = [
                t
                for t in purged
                if not (hi < idx[t] <= hi + embargo_bars) and not (lo - embargo_bars <= idx[t] < lo)
            ]
        if not purged:
            continue
        folds.append(Fold(train_times=purged, val_times=[], test_times=test_times))
    return folds


def _require_group_counts(n_groups: int, n_test_groups: int) -> None:
    if n_groups < 2 or n_test_groups < 1 or n_test_groups >= n_groups:
        raise ValueError("invalid CPCV group counts")


def cpcv_n_splits(n_groups: int, n_test_groups: int) -> int:
    """Number of train/test splits, ``C(N, k)``."""
    _require_group_counts(n_groups, n_test_groups)
    return int(comb(n_groups, n_test_groups))


def cpcv_n_paths(n_groups: int, n_test_groups: int) -> int:
    """Number of reconstructed backtest paths, ``C(N-1, k-1)``.

    López de Prado (AFML, 2018, ch. 12) shows this equals
    ``C(N, k) * k / N``: each group is tested in that many splits, and each
    path tests the group exactly once.
    """
    _require_group_counts(n_groups, n_test_groups)
    return int(comb(n_groups - 1, n_test_groups - 1))


def cpcv_path_assignments(n_groups: int, n_test_groups: int) -> IntArray:
    """Assign each group on each backtest path to one split that tests it.

    Returns an integer array of shape ``(n_paths, n_groups)``. Entry
    ``[p, g]`` is the index of a combination in lexicographic order (the
    same order as :func:`itertools.combinations`) whose test set contains
    group ``g``. Each path covers every group once. Each incidence
    ``(split, group)`` with ``group`` in the split appears on exactly one
    path, so stitched paths do not reuse a test forecast.
    """
    n_paths = cpcv_n_paths(n_groups, n_test_groups)
    splits = list(combinations(range(n_groups), n_test_groups))
    by_group: list[list[int]] = [[] for _ in range(n_groups)]
    for split_id, combo in enumerate(splits):
        for group in combo:
            by_group[group].append(split_id)
    assignments = np.empty((n_paths, n_groups), dtype=np.int64)
    for group, split_ids in enumerate(by_group):
        if len(split_ids) != n_paths:
            raise RuntimeError("CPCV path incidence is not balanced")
        assignments[:, group] = np.asarray(split_ids, dtype=np.int64)
    return assignments


def stitch_group_paths(
    group_scores: NDArray[np.float64], assignments: IntArray
) -> NDArray[np.float64]:
    """Stitch per-split group scores into backtest paths.

    ``group_scores`` has shape ``(n_splits, n_groups)`` and stores the
    out-of-sample score of each group under each split. Only entries whose
    group belongs to that split are read. Returns shape ``(n_paths, n_groups)``.
    """
    scores = np.asarray(group_scores, dtype=float)
    paths = np.asarray(assignments)
    if scores.ndim != 2 or paths.ndim != 2:
        raise ValueError("group_scores and assignments must be 2-d")
    if paths.shape[1] != scores.shape[1]:
        raise ValueError("assignments and group_scores disagree on n_groups")
    if paths.size and int(paths.max()) >= scores.shape[0]:
        raise ValueError("assignment split id is outside group_scores")
    n_paths, n_groups = paths.shape
    stitched = np.empty((n_paths, n_groups), dtype=float)
    for group in range(n_groups):
        stitched[:, group] = scores[paths[:, group], group]
    return stitched


def combinatorial_purged_indices(
    n: int,
    n_groups: int,
    n_test_groups: int,
    *,
    label_horizon: int,
    embargo: int,
) -> list[tuple[IntArray, IntArray]]:
    """Index-space CPCV: every ``C(N, k)`` split, purged by label horizon.

    Observations are the contiguous positions ``0 .. n-1``, partitioned into
    ``n_groups`` sequential groups. For each combination of ``n_test_groups``,
    the test set is the union of those groups and the train set is the
    complement after per-group purge and embargo.

    Purge drops a train index ``i`` whose label window ``(i, i+label_horizon]``
    reaches a test block. Embargo drops the ``embargo`` bars immediately
    before and after each test block. Folds whose train set is emptied are
    omitted (the same fail-closed rule as :func:`combinatorial_purged_cv`).

    The label horizon is measured in observation steps, matching a session
    index on a gap-free timeline.
    """
    _require_group_counts(n_groups, n_test_groups)
    if label_horizon < 0:
        raise ValueError("label_horizon must be non-negative")
    if embargo < 0:
        raise ValueError("embargo must be non-negative")
    if n < n_groups:
        raise ValueError(f"need at least n_groups={n_groups} observations for CPCV, got {n}")
    bounds = [int(i * n / n_groups) for i in range(n_groups + 1)]
    groups = [np.arange(bounds[i], bounds[i + 1], dtype=np.int64) for i in range(n_groups)]
    if any(group.size == 0 for group in groups):
        raise ValueError("CPCV group bounds produced an empty group")
    folds: list[tuple[IntArray, IntArray]] = []
    for test_ids in combinations(range(n_groups), n_test_groups):
        chosen = set(test_ids)
        test = np.concatenate([groups[group] for group in test_ids])
        train_parts = [groups[group] for group in range(n_groups) if group not in chosen]
        train = np.concatenate(train_parts) if train_parts else np.empty(0, dtype=np.int64)
        if test.size == 0 or train.size == 0:
            continue
        keep = np.ones(train.size, dtype=bool)
        for group in test_ids:
            block = groups[group]
            lo = int(block[0])
            hi = int(block[-1])
            idx = train
            in_holdout = (idx >= lo) & (idx <= hi)
            label_reaches = (idx < lo) & ((idx + label_horizon) >= lo)
            embargo_after = (idx > hi) & (idx <= hi + embargo)
            embargo_before = (idx >= lo - embargo) & (idx < lo)
            keep &= ~(in_holdout | label_reaches | embargo_after | embargo_before)
        kept = train[keep]
        if kept.size == 0:
            continue
        folds.append((kept.astype(np.int64, copy=False), test.astype(np.int64, copy=False)))
    return folds


def _contiguous_blocks(ids: list[int]) -> list[tuple[int, int]]:
    """Collapse sorted session ids into inclusive contiguous ``[lo, hi]`` blocks."""
    if not ids:
        return []
    ordered = sorted(ids)
    blocks: list[tuple[int, int]] = []
    lo = hi = ordered[0]
    for j in ordered[1:]:
        if j == hi + 1:
            hi = j
        else:
            blocks.append((lo, hi))
            lo = hi = j
    blocks.append((lo, hi))
    return blocks


def fold_embargo_report(
    folds: list[Fold],
    times: list[datetime],
    *,
    horizon_bars: int,
    embargo_bars: int,
) -> dict[str, Any]:
    """Bar-level purge/embargo evidence for datetime CPCV folds (SOTA G7).

    :func:`combinatorial_purged_cv` applies purge and embargo, but nothing
    previously *asserted at the timestamp level* that a fold really dropped
    ``>= embargo_bars`` sessions after each test block. Set-disjointness of
    train and test dates (what ``bench_cpcv_audit`` checks) is weaker: a fold
    can be disjoint and still train on the bar immediately after a test
    block, which is exactly the serial dependence the embargo exists to cut.

    For every fold and every **contiguous** test block ``[lo, hi]`` (session
    indices over ``times``) this counts violations of:

    1. ``train_in_test`` — a train session inside the inclusive block;
    2. ``label_reaches_test`` — a train label window ``(i, i+horizon_bars]``
       intersecting the block;
    3. ``embargo_after_violation`` — a train session in ``(hi, hi+embargo]``;
    4. ``embargo_before_violation`` — a train session in ``[lo-embargo, lo)``.

    Returns a dict whose ``ok`` is True only when all four counts are zero
    across all folds. ``n_blocks`` is the number of contiguous test blocks
    inspected, so a caller can tell "no violations" from "nothing inspected".
    Research-integrity diagnostic only; it carries no ratio or P&L key.

    Fail-closed: a fold timestamp absent from ``times``, negative
    horizon/embargo, or an empty test block raise ``ValueError``.
    """
    if horizon_bars < 0:
        raise ValueError("horizon_bars must be non-negative")
    if embargo_bars < 0:
        raise ValueError("embargo_bars must be non-negative")
    idx = session_index(times)
    train_in_test = 0
    label_reaches = 0
    embargo_after = 0
    embargo_before = 0
    n_blocks = 0
    for fold in folds:
        test_ids = []
        for t in fold.test_times:
            if t not in idx:
                raise ValueError(f"fold test timestamp {t} is not in times")
            test_ids.append(idx[t])
        if not test_ids:
            raise ValueError("fold has an empty test block")
        train_ids = []
        for t in fold.train_times:
            if t not in idx:
                raise ValueError(f"fold train timestamp {t} is not in times")
            train_ids.append(idx[t])
        train_set = set(train_ids)
        for lo, hi in _contiguous_blocks(test_ids):
            n_blocks += 1
            for i in train_ids:
                if lo <= i <= hi:
                    train_in_test += 1
                if horizon_bars > 0 and i < lo <= i + horizon_bars:
                    label_reaches += 1
            for j in range(hi + 1, hi + embargo_bars + 1):
                if j in train_set:
                    embargo_after += 1
            for j in range(lo - embargo_bars, lo):
                if j in train_set:
                    embargo_before += 1
    total = train_in_test + label_reaches + embargo_after + embargo_before
    return {
        "claim": "validation_integrity_only",
        "n_folds": len(folds),
        "n_blocks": n_blocks,
        "horizon_bars": int(horizon_bars),
        "embargo_bars": int(embargo_bars),
        "train_in_test": train_in_test,
        "label_reaches_test": label_reaches,
        "embargo_after_violation": embargo_after,
        "embargo_before_violation": embargo_before,
        "n_violations": total,
        "ok": total == 0 and n_blocks > 0,
    }


def assert_fold_embargo(
    folds: list[Fold],
    times: list[datetime],
    *,
    horizon_bars: int,
    embargo_bars: int,
) -> dict[str, Any]:
    """Raise ``AssertionError`` unless every fold honours purge and embargo.

    Enforcement wrapper over :func:`fold_embargo_report`: the report is
    *checked*, not merely annotated. Returns the report so a caller can stamp
    the evidence on a receipt after the assertion has passed.
    """
    report = fold_embargo_report(folds, times, horizon_bars=horizon_bars, embargo_bars=embargo_bars)
    if not report["ok"]:
        raise AssertionError(f"CPCV fold integrity violated: {report}")
    return report


def cpcv_folds_verified(
    times: list[datetime],
    n_groups: int,
    n_test_groups: int,
    horizon_bars: int,
    embargo_bars: int,
) -> tuple[list[Fold], dict[str, Any]]:
    """CPCV folds plus a bar-level integrity report that was actually checked.

    Same splits as :func:`combinatorial_purged_cv`, but the returned folds are
    asserted to honour purge and embargo at the timestamp level before they are
    handed back (SOTA G7). Raises ``AssertionError`` on any violation.
    """
    folds = combinatorial_purged_cv(times, n_groups, n_test_groups, horizon_bars, embargo_bars)
    report = assert_fold_embargo(folds, times, horizon_bars=horizon_bars, embargo_bars=embargo_bars)
    return folds, report


@dataclass(frozen=True)
class ConfigRanking:
    """Conservative ranking of configs over reconstructed CPCV paths.

    ``selected`` is the winner under ``criterion``; ``order`` is best-first.
    ``detail`` maps each config to its ``mean``, ``worst_path``,
    ``path_dispersion`` (max - min across paths) and ``n_paths``.

    Research-integrity diagnostic only (``claim``): it ranks **proper scores**
    (IC, pinball, CRPS — whatever the caller stitched onto the paths) and never
    produces a ratio key. It does not promote anything and does not move
    ``blend_weight``.
    """

    criterion: str
    selected: str | None
    order: tuple[str, ...]
    detail: dict[str, dict[str, float]]
    claim: str = "research_diagnostic_only"
    research_only: bool = True


def rank_configs(
    path_scores: dict[str, NDArray[np.float64]],
    *,
    criterion: RankCriterion = "mean",
) -> ConfigRanking:
    """Rank configs by mean path score or by their **worst** path score.

    ``path_scores[name]`` is either ``(n_paths,)`` — one already-reduced score
    per reconstructed path — or ``(n_paths, n_periods)``, in which case each
    path is reduced to its mean first. Higher is better (feed IC or a negated
    loss, never a ratio). All configs must carry the same number of paths.

    ``criterion="mean"`` reproduces the naive average-over-paths selection:
    one lucky path can carry a config. ``criterion="worst_path"`` ranks on
    ``min`` over paths, so a config that only works on a single split is
    ranked by the split where it fails — the conservative criterion the
    overfitting literature asks for. The default stays ``"mean"`` so no
    existing caller's behaviour changes silently.

    Ties break deterministically on the config name. Fail-closed: an unknown
    criterion, an empty mapping, a non-1-D/2-D or empty array, mismatched path
    counts, or any non-finite score raise ``ValueError`` — a ranking that
    silently dropped a config would understate the multiplicity.
    """
    if criterion not in RANK_CRITERIA:
        raise ValueError(f"criterion must be one of {RANK_CRITERIA}, got {criterion!r}")
    if not path_scores:
        raise ValueError("path_scores must not be empty")
    reduced: dict[str, NDArray[np.float64]] = {}
    n_paths: int | None = None
    for name, raw in path_scores.items():
        key = str(name)
        if not key:
            raise ValueError("config name must be non-empty")
        if key in reduced:
            raise ValueError(f"duplicate config name: {key!r}")
        arr = np.asarray(raw, dtype=float)
        if arr.ndim == 2:
            if arr.shape[0] == 0 or arr.shape[1] == 0:
                raise ValueError(f"config {key!r} has an empty path panel")
            arr = arr.mean(axis=1)
        elif arr.ndim != 1:
            raise ValueError(f"config {key!r} scores must be 1-d or 2-d, got ndim={arr.ndim}")
        if arr.size == 0:
            raise ValueError(f"config {key!r} has no path scores")
        if not bool(np.isfinite(arr).all()):
            raise ValueError(f"config {key!r} path scores must be finite")
        if n_paths is None:
            n_paths = int(arr.size)
        elif int(arr.size) != n_paths:
            raise ValueError(
                f"config {key!r} has {arr.size} paths, expected {n_paths}; "
                "every config must be scored on the same reconstructed paths"
            )
        reduced[key] = arr
    detail: dict[str, dict[str, float]] = {}
    for key, arr in reduced.items():
        detail[key] = {
            "mean": float(np.mean(arr)),
            "worst_path": float(np.min(arr)),
            "best_path": float(np.max(arr)),
            "path_dispersion": float(np.max(arr) - np.min(arr)),
            "n_paths": float(arr.size),
        }
    ranked = sorted(
        reduced,
        key=lambda k: (-detail[k][criterion], k),
    )
    return ConfigRanking(
        criterion=str(criterion),
        selected=ranked[0] if ranked else None,
        order=tuple(ranked),
        detail=detail,
    )
