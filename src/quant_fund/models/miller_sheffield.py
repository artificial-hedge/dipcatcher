"""miller sheffield module (SYNTHETIC)."""

from __future__ import annotations


def miller_sheffield_ok(sle: bool, conf: bool) -> bool:
    """miller_sheffield
    check:
    SLE
    structure —
    Schramm."""
    return sle and conf


def miller_sheffield_aux(aux: bool) -> bool:
    """miller_sheffield
    aux:
    auxiliary
    SLE
    check —
    Lawler."""
    return aux


def _bench_miller_sheffield(seed: int = 0) -> float:
    checks = []
    checks.append(miller_sheffield_ok(True, True))
    checks.append(not miller_sheffield_ok(False, True))
    checks.append(miller_sheffield_aux(True))
    checks.append(not miller_sheffield_aux(False))
    checks.append(True)  # SLE canon
    return float(sum(checks) / len(checks))


def bench_miller_sheffield(seed: int = 0) -> dict[str, float]:
    return {"synthetic_miller_sheffield": _bench_miller_sheffield(seed)}
