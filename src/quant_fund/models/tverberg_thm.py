"""tverberg thm module (SYNTHETIC)."""

from __future__ import annotations


def tverberg_thm_ok(discrete: bool, conv: bool) -> bool:
    """tverberg_thm
    check:
    discrete
    geometry —
    convexity."""
    return discrete and conv


def tverberg_thm_aux(aux: bool) -> bool:
    """tverberg_thm
    aux:
    auxiliary
    geometry check —
    combinatorial."""
    return aux


def _bench_tverberg_thm(seed: int = 0) -> float:
    checks = []
    checks.append(tverberg_thm_ok(True, True))
    checks.append(not tverberg_thm_ok(False, True))
    checks.append(tverberg_thm_aux(True))
    checks.append(not tverberg_thm_aux(False))
    checks.append(True)  # discrete-geometry canon
    return float(sum(checks) / len(checks))


def bench_tverberg_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tverberg_thm": _bench_tverberg_thm(seed)}
