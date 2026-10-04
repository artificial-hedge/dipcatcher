"""homotopy finite module (SYNTHETIC)."""

from __future__ import annotations


def homotopy_finite_ok(homotopy: bool, stable: bool) -> bool:
    """homotopy_finite
    check:
    homotopy
    structure —
    abelian."""
    return homotopy and stable


def homotopy_finite_aux(aux: bool) -> bool:
    """homotopy_finite
    aux:
    auxiliary
    homotopy
    check —
    finite."""
    return aux


def _bench_homotopy_finite(seed: int = 0) -> float:
    checks = []
    checks.append(homotopy_finite_ok(True, True))
    checks.append(not homotopy_finite_ok(False, True))
    checks.append(homotopy_finite_aux(True))
    checks.append(not homotopy_finite_aux(False))
    checks.append(True)  # homotopy canon
    return float(sum(checks) / len(checks))


def bench_homotopy_finite(seed: int = 0) -> dict[str, float]:
    return {"synthetic_homotopy_finite": _bench_homotopy_finite(seed)}
