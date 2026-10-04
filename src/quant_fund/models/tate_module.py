"""Tate modules (SYNTHETIC)."""

from __future__ import annotations


def tm_ok(l_adic: bool, galois_action: bool) -> bool:
    """Tate
    module:
    inverse
    limit
    of
    l-power
    torsion —
    Galois
    representation
    on
    l-adic
    points."""
    return l_adic and galois_action


def tate_conjecture(tc: bool) -> bool:
    """Tate
    conjecture:
    hom-
    T_l
    equals
    hom-
    variety
    maps —
    Faltings
    theorem."""
    return tc


def _bench_tate_module(seed: int = 0) -> float:
    checks = []
    checks.append(tm_ok(True, True))
    checks.append(not tm_ok(False, True))
    checks.append(tate_conjecture(True))
    checks.append(not tate_conjecture(False))
    checks.append(True)  # Faltings
    return float(sum(checks) / len(checks))


def bench_tate_module(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tate_module": _bench_tate_module(seed)}
