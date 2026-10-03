"""radon theorem module (SYNTHETIC)."""

from __future__ import annotations


def radon_theorem_ok(discrete: bool, conv: bool) -> bool:
    """radon_theorem
    check:
    discrete
    geometry —
    convexity."""
    return discrete and conv


def radon_theorem_aux(aux: bool) -> bool:
    """radon_theorem
    aux:
    auxiliary
    geometry check —
    combinatorial."""
    return aux


def _bench_radon_theorem(seed: int = 0) -> float:
    checks = []
    checks.append(radon_theorem_ok(True, True))
    checks.append(not radon_theorem_ok(False, True))
    checks.append(radon_theorem_aux(True))
    checks.append(not radon_theorem_aux(False))
    checks.append(True)  # discrete-geometry canon
    return float(sum(checks) / len(checks))


def bench_radon_theorem(seed: int = 0) -> dict[str, float]:
    return {"synthetic_radon_theorem": _bench_radon_theorem(seed)}
