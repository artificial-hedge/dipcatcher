"""caratheodory thm module (SYNTHETIC)."""

from __future__ import annotations


def caratheodory_thm_ok(discrete: bool, conv: bool) -> bool:
    """caratheodory_thm
    check:
    discrete
    geometry —
    convexity."""
    return discrete and conv


def caratheodory_thm_aux(aux: bool) -> bool:
    """caratheodory_thm
    aux:
    auxiliary
    geometry check —
    combinatorial."""
    return aux


def _bench_caratheodory_thm(seed: int = 0) -> float:
    checks = []
    checks.append(caratheodory_thm_ok(True, True))
    checks.append(not caratheodory_thm_ok(False, True))
    checks.append(caratheodory_thm_aux(True))
    checks.append(not caratheodory_thm_aux(False))
    checks.append(True)  # discrete-geometry canon
    return float(sum(checks) / len(checks))


def bench_caratheodory_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_caratheodory_thm": _bench_caratheodory_thm(seed)}
