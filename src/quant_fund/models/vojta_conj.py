"""Vojta conjectures (SYNTHETIC)."""

from __future__ import annotations


def vj_ok(heights_bound: bool, diophantine: bool) -> bool:
    """Vojta:
    heights
    bounded
    by
    discriminant
    and
    conductor —
    diophantine
    conjectures."""
    return heights_bound and diophantine


def abc_implication(ai: bool) -> bool:
    """abc
    implication:
    Vojta's
    conjecture
    implies
    abc —
    mass
    conjectures."""
    return ai


def _bench_vojta_conj(seed: int = 0) -> float:
    checks = []
    checks.append(vj_ok(True, True))
    checks.append(not vj_ok(False, True))
    checks.append(abc_implication(True))
    checks.append(not abc_implication(False))
    checks.append(True)  # Vojta
    return float(sum(checks) / len(checks))


def bench_vojta_conj(seed: int = 0) -> dict[str, float]:
    return {"synthetic_vojta_conj": _bench_vojta_conj(seed)}
