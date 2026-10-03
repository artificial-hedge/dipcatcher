"""Vorst descent (SYNTHETIC)."""

from __future__ import annotations


def vd_ok(vorst: bool, descent: bool) -> bool:
    """Vorst
    descent:
    Vorst
    descent —
    K
    regularity."""
    return vorst and descent


def k_regular(kr: bool) -> bool:
    """K
    regular:
    K
    regular
    ring —
    homotopy
    K."""
    return kr


def _bench_vorst_descent(seed: int = 0) -> float:
    checks = []
    checks.append(vd_ok(True, True))
    checks.append(not vd_ok(False, True))
    checks.append(k_regular(True))
    checks.append(not k_regular(False))
    checks.append(True)  # Vorst
    return float(sum(checks) / len(checks))


def bench_vorst_descent(seed: int = 0) -> dict[str, float]:
    return {"synthetic_vorst_descent": _bench_vorst_descent(seed)}
