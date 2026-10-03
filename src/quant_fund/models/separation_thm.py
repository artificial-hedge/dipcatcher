"""separation thm module (SYNTHETIC)."""

from __future__ import annotations


def separation_thm_ok(discrete: bool, conv: bool) -> bool:
    """separation_thm
    check:
    discrete
    geometry —
    convexity."""
    return discrete and conv


def separation_thm_aux(aux: bool) -> bool:
    """separation_thm
    aux:
    auxiliary
    geometry check —
    combinatorial."""
    return aux


def _bench_separation_thm(seed: int = 0) -> float:
    checks = []
    checks.append(separation_thm_ok(True, True))
    checks.append(not separation_thm_ok(False, True))
    checks.append(separation_thm_aux(True))
    checks.append(not separation_thm_aux(False))
    checks.append(True)  # discrete-geometry canon
    return float(sum(checks) / len(checks))


def bench_separation_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_separation_thm": _bench_separation_thm(seed)}
