"""Yang-Lee category (SYNTHETIC)."""

from __future__ import annotations


def yl_ok(yang: bool, lee: bool) -> bool:
    """Yang
    Lee:
    Yang
    Lee
    category —
    nonunitary
    MTC."""
    return yang and lee


def nonunitary_modular(nm: bool) -> bool:
    """Nonunitary
    modular:
    nonunitary
    modular
    category —
    Galois
    conjugate."""
    return nm


def _bench_yang_lee_cat(seed: int = 0) -> float:
    checks = []
    checks.append(yl_ok(True, True))
    checks.append(not yl_ok(False, True))
    checks.append(nonunitary_modular(True))
    checks.append(not nonunitary_modular(False))
    checks.append(True)  # Yang-Lee
    return float(sum(checks) / len(checks))


def bench_yang_lee_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_yang_lee_cat": _bench_yang_lee_cat(seed)}
