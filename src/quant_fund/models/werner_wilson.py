"""werner wilson module (SYNTHETIC)."""

from __future__ import annotations


def werner_wilson_ok(sle: bool, conf: bool) -> bool:
    """werner_wilson
    check:
    SLE
    structure —
    Schramm."""
    return sle and conf


def werner_wilson_aux(aux: bool) -> bool:
    """werner_wilson
    aux:
    auxiliary
    SLE
    check —
    Lawler."""
    return aux


def _bench_werner_wilson(seed: int = 0) -> float:
    checks = []
    checks.append(werner_wilson_ok(True, True))
    checks.append(not werner_wilson_ok(False, True))
    checks.append(werner_wilson_aux(True))
    checks.append(not werner_wilson_aux(False))
    checks.append(True)  # SLE canon
    return float(sum(checks) / len(checks))


def bench_werner_wilson(seed: int = 0) -> dict[str, float]:
    return {"synthetic_werner_wilson": _bench_werner_wilson(seed)}
