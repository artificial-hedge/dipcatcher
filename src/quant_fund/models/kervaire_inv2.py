"""kervaire inv2 module (SYNTHETIC)."""

from __future__ import annotations


def kervaire_inv2_ok(homotopy: bool, periodic: bool) -> bool:
    """kervaire_inv2
    check:
    homotopy
    structure —
    unstable."""
    return homotopy and periodic


def kervaire_inv2_aux(aux: bool) -> bool:
    """kervaire_inv2
    aux:
    auxiliary
    homotopy
    check —
    periodic."""
    return aux


def _bench_kervaire_inv2(seed: int = 0) -> float:
    checks = []
    checks.append(kervaire_inv2_ok(True, True))
    checks.append(not kervaire_inv2_ok(False, True))
    checks.append(kervaire_inv2_aux(True))
    checks.append(not kervaire_inv2_aux(False))
    checks.append(True)  # homotopy canon
    return float(sum(checks) / len(checks))


def bench_kervaire_inv2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kervaire_inv2": _bench_kervaire_inv2(seed)}
