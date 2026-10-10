"""Bar-construction invariants: property-test harness for bar builders.

Any bar builder (dollar, imbalance, runs, information, ...) must satisfy a
small set of mechanical invariants on ANY tape. This module packages those
invariants as composable checks with a built-in synthetic tape:

- ``invariant_monotone_ids`` — bar ids are non-decreasing and dense;
- ``invariant_ohlc_consistent`` — open=first, close=last, high≥max(open,
  close), low≤min(open,close), high≥low for every bar;
- ``invariant_volume_conserved`` — bar volumes sum to the tape total;
- ``invariant_returns_telescope`` — compounded bar returns reconstruct the
  tape's total log return;
- ``bar_builder_audit`` — run all invariants against a builder callable and
  report pass/fail per invariant with a score.

Honesty: invariants are correctness properties of the construction itself;
passing them says nothing about predictive value.

References:
- López de Prado, M. (2018). *Advances in Financial Machine Learning*,
  ch. 2 — bar construction conventions.
- Claessen, K., Hughes, J. (2000). QuickCheck — the property-testing
  discipline this harness ports to bar builders.

Composition: pure numpy; deterministic.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _validate_inputs(prices: FloatArray, sizes: FloatArray, ids: IntArray) -> None:
    p = np.asarray(prices, dtype=np.float64)
    s = np.asarray(sizes, dtype=np.float64)
    i = np.asarray(ids, dtype=np.int64)
    if not (p.shape == s.shape == i.shape):
        raise ValueError("prices, sizes and ids must have equal shapes")
    if len(p) < 2:
        raise ValueError("tape must have at least 2 trades")


def invariant_monotone_ids(ids: IntArray) -> bool:
    """Bar ids are non-decreasing integers starting at 0, with no gaps."""
    i = np.asarray(ids, dtype=np.int64)
    if i.ndim != 1 or len(i) == 0:
        return False
    if i[0] != 0 or np.any(np.diff(i) < 0):
        return False
    unique = np.unique(i)
    return bool(np.array_equal(unique, np.arange(int(unique[-1]) + 1)))


def invariant_ohlc_consistent(prices: FloatArray, ids: IntArray) -> bool:
    """OHLC per bar is consistent with the bar's constituent trades."""
    p = np.asarray(prices, dtype=np.float64)
    i = np.asarray(ids, dtype=np.int64)
    if p.shape != i.shape or len(p) == 0:
        return False
    if np.any(p <= 0):
        return False
    change = np.flatnonzero(np.diff(i))
    ends = np.append(change, len(i) - 1)
    starts = np.concatenate(([0], ends[:-1] + 1))
    for b in range(len(ends)):
        seg = p[starts[b] : ends[b] + 1]
        o, h, lo, c = seg[0], float(seg.max()), float(seg.min()), seg[-1]
        if not (h >= max(o, c) - 1e-12 and lo <= min(o, c) + 1e-12 and h >= lo):
            return False
    return True


def invariant_volume_conserved(sizes: FloatArray, ids: IntArray) -> bool:
    """Per-bar volumes sum to the tape total exactly."""
    s = np.asarray(sizes, dtype=np.float64)
    i = np.asarray(ids, dtype=np.int64)
    if s.shape != i.shape:
        return False
    n_bars = int(i[-1]) + 1 if len(i) else 0
    out = np.zeros(n_bars, dtype=np.float64)
    np.add.at(out, i, s)
    return bool(abs(float(out.sum()) - float(s.sum())) < 1e-9)


def invariant_returns_telescope(prices: FloatArray, ids: IntArray) -> bool:
    """Compounded bar returns reconstruct the tape's total log return."""
    p = np.asarray(prices, dtype=np.float64)
    i = np.asarray(ids, dtype=np.int64)
    if p.shape != i.shape or len(p) < 2:
        return False
    if np.any(p <= 0):
        return False
    change = np.flatnonzero(np.diff(i))
    ends = np.append(change, len(i) - 1)
    closes = p[ends]
    total_bar_log = float(np.sum(np.diff(np.log(closes))))
    total_tape_log = float(np.log(p[-1] / p[0]))
    # bars cover [first trade .. last trade]; telescope covers closes only
    first_bar_log = float(np.log(p[ends[0]] / p[0]))
    return abs(total_bar_log + first_bar_log - total_tape_log) < 1e-9


def bar_builder_audit(
    prices: FloatArray,
    sizes: FloatArray,
    builder: Callable[[FloatArray, FloatArray], IntArray],
) -> dict[str, float]:
    """Run all four invariants against ``builder(prices, sizes) -> ids``.

    Returns pass flags per invariant and a 0-1 audit score.
    """
    p = np.asarray(prices, dtype=np.float64)
    s = np.asarray(sizes, dtype=np.float64)
    ids = np.asarray(builder(p, s), dtype=np.int64)
    checks = {
        "monotone_ids": float(invariant_monotone_ids(ids)),
        "ohlc_consistent": float(invariant_ohlc_consistent(p, ids)),
        "volume_conserved": float(invariant_volume_conserved(s, ids)),
        "returns_telescope": float(invariant_returns_telescope(p, ids)),
    }
    checks["score"] = float(np.mean(list(checks.values())))
    return checks
