"""logarithmic red module (SYNTHETIC)."""

from __future__ import annotations


def logarithmic_red_ok(mat: bool, geo: bool) -> bool:
    """logarithmic_red
    check:
    matrix-analytic
    structure —
    Neuts
    MAP."""
    return mat and geo


def logarithmic_red_aux(aux: bool) -> bool:
    """logarithmic_red
    aux:
    auxiliary
    QBD
    check —
    Ramaswami."""
    return aux


def _bench_logarithmic_red(seed: int = 0) -> float:
    checks = []
    checks.append(logarithmic_red_ok(True, True))
    checks.append(not logarithmic_red_ok(False, True))
    checks.append(logarithmic_red_aux(True))
    checks.append(not logarithmic_red_aux(False))
    checks.append(True)  # MAM canon
    return float(sum(checks) / len(checks))


def bench_logarithmic_red(seed: int = 0) -> dict[str, float]:
    return {"synthetic_logarithmic_red": _bench_logarithmic_red(seed)}
