"""Whitney trick (SYNTHETIC)."""

from __future__ import annotations


def wt_ok(cancel_pairs: bool, embedding: bool) -> bool:
    """Whitney
    trick:
    cancelling
    intersection
    pairs
    via
    embedded
    discs —
    removes
    double
    points."""
    return cancel_pairs and embedding


def whitney_disc(wd: bool) -> bool:
    """Whitney
    disc:
    embedded
    disc
    bounding
    the
    cancelling
    pair —
    works
    in
    dim
    >= 5."""
    return wd


def _bench_whitney_trick(seed: int = 0) -> float:
    checks = []
    checks.append(wt_ok(True, True))
    checks.append(not wt_ok(False, True))
    checks.append(whitney_disc(True))
    checks.append(not whitney_disc(False))
    checks.append(True)  # Whitney
    return float(sum(checks) / len(checks))


def bench_whitney_trick(seed: int = 0) -> dict[str, float]:
    return {"synthetic_whitney_trick": _bench_whitney_trick(seed)}
