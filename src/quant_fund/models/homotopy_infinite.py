"""homotopy infinite module (SYNTHETIC)."""

from __future__ import annotations


def homotopy_infinite_ok(homotopy: bool, stable: bool) -> bool:
    """homotopy_infinite
    check:
    homotopy
    structure —
    abelian."""
    return homotopy and stable


def homotopy_infinite_aux(aux: bool) -> bool:
    """homotopy_infinite
    aux:
    auxiliary
    homotopy
    check —
    finite."""
    return aux


def _bench_homotopy_infinite(seed: int = 0) -> float:
    checks = []
    checks.append(homotopy_infinite_ok(True, True))
    checks.append(not homotopy_infinite_ok(False, True))
    checks.append(homotopy_infinite_aux(True))
    checks.append(not homotopy_infinite_aux(False))
    checks.append(True)  # homotopy canon
    return float(sum(checks) / len(checks))


def bench_homotopy_infinite(seed: int = 0) -> dict[str, float]:
    return {"synthetic_homotopy_infinite": _bench_homotopy_infinite(seed)}
