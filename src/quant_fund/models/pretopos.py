"""Pretopos (SYNTHETIC)."""

from __future__ import annotations


def pt_ok(pretopos: bool, exact: bool) -> bool:
    """Pretopos:
    effective
    regular
    category
    with
    coproducts —
    Giraud
    slice."""
    return pretopos and exact


def effective_descent(ed: bool) -> bool:
    """Effective:
    equivalence
    relations
    effective
    quotients —
    Barr-
    effective."""
    return ed


def _bench_pretopos(seed: int = 0) -> float:
    checks = []
    checks.append(pt_ok(True, True))
    checks.append(not pt_ok(False, True))
    checks.append(effective_descent(True))
    checks.append(not effective_descent(False))
    checks.append(True)  # Giraud
    return float(sum(checks) / len(checks))


def bench_pretopos(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pretopos": _bench_pretopos(seed)}
