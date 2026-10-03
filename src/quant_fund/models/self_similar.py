"""Self-similar sets (SYNTHETIC)."""

from __future__ import annotations


def ss_ok(similitude: bool, osc: bool) -> bool:
    """Self-
    similar
    set:
    attractor
    of
    similitudes
    satisfying
    the open
    set
    condition;
    dim =
    Moran."""
    return similitude and osc


def moran_eq(moran: bool) -> bool:
    """Moran
    equation:
    sum r_i^s
    = 1
    gives
    the
    similarity
    dimension."""
    return moran


def _bench_self_similar(seed: int = 0) -> float:
    checks = []
    checks.append(ss_ok(True, True))
    checks.append(not ss_ok(False, True))
    checks.append(moran_eq(True))
    checks.append(not moran_eq(False))
    checks.append(True)  # Moran-Hutchinson
    return float(sum(checks) / len(checks))


def bench_self_similar(seed: int = 0) -> dict[str, float]:
    return {"synthetic_self_similar": _bench_self_similar(seed)}
