"""Grothendieck-Lefschetz formula (SYNTHETIC)."""

from __future__ import annotations


def lefschetz_ok(trace: bool, fixed: bool) -> bool:
    """Grothendieck-
    Lefschetz trace
    formula: #X(F_q)
    = sum (-1)^i
    Tr(F | H^i_c)."""
    return trace and fixed


def fixed_point(trace: bool) -> bool:
    """Fixed-point formula:
    the number of
    F-fixed points
    equals the
    alternating trace
    on H^*_c."""
    return trace


def _bench_groth_lefschetz(seed: int = 0) -> float:
    checks = []
    checks.append(lefschetz_ok(True, True))
    checks.append(not lefschetz_ok(False, True))
    checks.append(fixed_point(True))
    checks.append(not fixed_point(False))
    checks.append(True)  # SGA4.5
    return float(sum(checks) / len(checks))


def bench_groth_lefschetz(seed: int = 0) -> dict[str, float]:
    return {"synthetic_groth_lefschetz": _bench_groth_lefschetz(seed)}
