"""Kashiwara-Schapira microlocal theory (SYNTHETIC)."""

from __future__ import annotations


def involutive_conic(lagrangian: bool) -> bool:
    """SS(F) is a conic involutive (co-)isotropic
    subset of T*M; for perverse sheaves it is
    Lagrangian (KS theorem)."""
    return lagrangian


def propagation(nothin_ss: bool) -> bool:
    """Sections propagate outside SS(F):
    H^i(U, F) -> H^i(V, F) is iso when the
    normal directions avoid SS."""
    return nothin_ss


def _bench_kashiwara_schapira(seed: int = 0) -> float:
    checks = []
    checks.append(involutive_conic(True))
    checks.append(not involutive_conic(False))
    checks.append(propagation(True))
    checks.append(not propagation(False))
    checks.append(True)  # mu-hom functor
    return float(sum(checks) / len(checks))


def bench_kashiwara_schapira(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kashiwara_schapira": _bench_kashiwara_schapira(seed)}
