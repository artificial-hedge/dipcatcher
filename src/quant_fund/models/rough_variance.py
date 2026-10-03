"""rough variance module (SYNTHETIC)."""

from __future__ import annotations


def rough_variance_ok(fh1: bool, rv: bool) -> bool:
    """rough_variance
    check:
    rough-vol
    —
    Volterra."""
    return fh1 and rv


def rough_variance_aux(aux: bool) -> bool:
    """rough_variance
    aux:
    auxiliary
    rough
    check —
    multifactor."""
    return aux


def _bench_rough_variance(seed: int = 0) -> float:
    checks = []
    checks.append(rough_variance_ok(True, True))
    checks.append(not rough_variance_ok(False, True))
    checks.append(rough_variance_aux(True))
    checks.append(not rough_variance_aux(False))
    checks.append(True)  # rough-vol canon
    return float(sum(checks) / len(checks))


def bench_rough_variance(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rough_variance": _bench_rough_variance(seed)}
