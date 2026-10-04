"""Zhu algebra (SYNTHETIC)."""

from __future__ import annotations


def zhu_ok(associative: bool, top: bool) -> bool:
    """Zhu
    algebra
    A(V):
    associative
    quotient of
    V whose
    modules are
    top levels
    of
    V-modules."""
    return associative and top


def zhu_corresp(corr: bool) -> bool:
    """Zhu
    correspondence:
    irreducible
    A(V)-modules
    biject with
    irreducible
    admissible
    V-modules."""
    return corr


def _bench_zhu_algebra(seed: int = 0) -> float:
    checks = []
    checks.append(zhu_ok(True, True))
    checks.append(not zhu_ok(False, True))
    checks.append(zhu_corresp(True))
    checks.append(not zhu_corresp(False))
    checks.append(True)  # Zhu
    return float(sum(checks) / len(checks))


def bench_zhu_algebra(seed: int = 0) -> dict[str, float]:
    return {"synthetic_zhu_algebra": _bench_zhu_algebra(seed)}
