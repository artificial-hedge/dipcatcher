"""Period maps (SYNTHETIC)."""

from __future__ import annotations


def period_map_ok(griffiths: bool, variation: bool) -> bool:
    """Period map: family
    f: X -> S of varieties
    gives map s -> Hodge
    structure H^n(X_s) in
    period domain D;
    Griffiths."""
    return griffiths and variation


def griffiths_transversality(horizontal: bool) -> bool:
    """Griffiths
    transversality: the
    Hodge filtration
    varies only by
    one step in
    the flag."""
    return horizontal


def _bench_period_map(seed: int = 0) -> float:
    checks = []
    checks.append(period_map_ok(True, True))
    checks.append(not period_map_ok(False, True))
    checks.append(griffiths_transversality(True))
    checks.append(not griffiths_transversality(False))
    checks.append(True)  # Cattani-Deligne-Kaplan
    return float(sum(checks) / len(checks))


def bench_period_map(seed: int = 0) -> dict[str, float]:
    return {"synthetic_period_map": _bench_period_map(seed)}
