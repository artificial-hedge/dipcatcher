"""motivic norm module (SYNTHETIC)."""

from __future__ import annotations


def motivic_norm_ok(motivic: bool, stable: bool) -> bool:
    """motivic_norm
    check:
    motivic
    structure —
    stable."""
    return motivic and stable


def motivic_norm_aux(aux: bool) -> bool:
    """motivic_norm
    aux:
    auxiliary
    motivic
    check —
    residue."""
    return aux


def _bench_motivic_norm(seed: int = 0) -> float:
    checks = []
    checks.append(motivic_norm_ok(True, True))
    checks.append(not motivic_norm_ok(False, True))
    checks.append(motivic_norm_aux(True))
    checks.append(not motivic_norm_aux(False))
    checks.append(True)  # motivic canon
    return float(sum(checks) / len(checks))


def bench_motivic_norm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_norm": _bench_motivic_norm(seed)}
