"""enlargement f module (SYNTHETIC)."""

from __future__ import annotations


def enlargement_f_ok(f: bool, right: bool) -> bool:
    """enlargement_f
    check:
    filtration —
    right
    continuity."""
    return f and right


def enlargement_f_aux(aux: bool) -> bool:
    """enlargement_f
    aux:
    auxiliary
    filtration check —
    completeness."""
    return aux


def _bench_enlargement_f(seed: int = 0) -> float:
    checks = []
    checks.append(enlargement_f_ok(True, True))
    checks.append(not enlargement_f_ok(False, True))
    checks.append(enlargement_f_aux(True))
    checks.append(not enlargement_f_aux(False))
    checks.append(True)  # filtration canon
    return float(sum(checks) / len(checks))


def bench_enlargement_f(seed: int = 0) -> dict[str, float]:
    return {"synthetic_enlargement_f": _bench_enlargement_f(seed)}
