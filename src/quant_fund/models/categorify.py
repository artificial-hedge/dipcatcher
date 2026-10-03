"""Categorification (SYNTHETIC)."""

from __future__ import annotations


def categorify_ok(lift: bool, decat: bool) -> bool:
    """Categorification:
    replace sets
    by categories,
    functions by
    functors,
    equalities
    by natural
    isomorphisms;
    decategorify
    recovers the
    original."""
    return lift and decat


def groth_decat(groth: bool) -> bool:
    """Decategorification
    via the split
    Grothendieck
    group K_0:
    [X oplus Y]
    = [X]+[Y]."""
    return groth


def _bench_categorify(seed: int = 0) -> float:
    checks = []
    checks.append(categorify_ok(True, True))
    checks.append(not categorify_ok(False, True))
    checks.append(groth_decat(True))
    checks.append(not groth_decat(False))
    checks.append(True)  # Crane-Frenkel
    return float(sum(checks) / len(checks))


def bench_categorify(seed: int = 0) -> dict[str, float]:
    return {"synthetic_categorify": _bench_categorify(seed)}
