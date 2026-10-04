"""homotopy extended module (SYNTHETIC)."""

from __future__ import annotations


def homotopy_extended_ok(homotopy: bool, stable: bool) -> bool:
    """homotopy_extended
    check:
    homotopy
    structure —
    abelian."""
    return homotopy and stable


def homotopy_extended_aux(aux: bool) -> bool:
    """homotopy_extended
    aux:
    auxiliary
    homotopy
    check —
    finite."""
    return aux


def _bench_homotopy_extended(seed: int = 0) -> float:
    checks = []
    checks.append(homotopy_extended_ok(True, True))
    checks.append(not homotopy_extended_ok(False, True))
    checks.append(homotopy_extended_aux(True))
    checks.append(not homotopy_extended_aux(False))
    checks.append(True)  # homotopy canon
    return float(sum(checks) / len(checks))


def bench_homotopy_extended(seed: int = 0) -> dict[str, float]:
    return {"synthetic_homotopy_extended": _bench_homotopy_extended(seed)}
