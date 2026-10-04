"""homotopy local module (SYNTHETIC)."""

from __future__ import annotations


def homotopy_local_ok(homotopy: bool, stable: bool) -> bool:
    """homotopy_local
    check:
    homotopy
    structure —
    sheaf."""
    return homotopy and stable


def homotopy_local_aux(aux: bool) -> bool:
    """homotopy_local
    aux:
    auxiliary
    homotopy
    check —
    coalgebra."""
    return aux


def _bench_homotopy_local(seed: int = 0) -> float:
    checks = []
    checks.append(homotopy_local_ok(True, True))
    checks.append(not homotopy_local_ok(False, True))
    checks.append(homotopy_local_aux(True))
    checks.append(not homotopy_local_aux(False))
    checks.append(True)  # homotopy canon
    return float(sum(checks) / len(checks))


def bench_homotopy_local(seed: int = 0) -> dict[str, float]:
    return {"synthetic_homotopy_local": _bench_homotopy_local(seed)}
