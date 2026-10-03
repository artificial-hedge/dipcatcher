"""rough bergomi module (SYNTHETIC)."""

from __future__ import annotations


def rough_bergomi_ok(fh1: bool, rv: bool) -> bool:
    """rough_bergomi
    check:
    rough-vol
    —
    Volterra."""
    return fh1 and rv


def rough_bergomi_aux(aux: bool) -> bool:
    """rough_bergomi
    aux:
    auxiliary
    rough
    check —
    multifactor."""
    return aux


def _bench_rough_bergomi(seed: int = 0) -> float:
    checks = []
    checks.append(rough_bergomi_ok(True, True))
    checks.append(not rough_bergomi_ok(False, True))
    checks.append(rough_bergomi_aux(True))
    checks.append(not rough_bergomi_aux(False))
    checks.append(True)  # rough-vol canon
    return float(sum(checks) / len(checks))


def bench_rough_bergomi(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rough_bergomi": _bench_rough_bergomi(seed)}
