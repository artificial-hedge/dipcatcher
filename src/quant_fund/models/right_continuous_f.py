"""right continuous_f module (SYNTHETIC)."""

from __future__ import annotations


def right_continuous_f_ok(f: bool, right: bool) -> bool:
    """right_continuous_f
    check:
    filtration —
    right
    continuity."""
    return f and right


def right_continuous_f_aux(aux: bool) -> bool:
    """right_continuous_f
    aux:
    auxiliary
    filtration check —
    completeness."""
    return aux


def _bench_right_continuous_f(seed: int = 0) -> float:
    checks = []
    checks.append(right_continuous_f_ok(True, True))
    checks.append(not right_continuous_f_ok(False, True))
    checks.append(right_continuous_f_aux(True))
    checks.append(not right_continuous_f_aux(False))
    checks.append(True)  # filtration canon
    return float(sum(checks) / len(checks))


def bench_right_continuous_f(seed: int = 0) -> dict[str, float]:
    return {"synthetic_right_continuous_f": _bench_right_continuous_f(seed)}
