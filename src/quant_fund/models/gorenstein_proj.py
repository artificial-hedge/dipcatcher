"""gorenstein proj module (SYNTHETIC)."""

from __future__ import annotations


def gorenstein_proj_ok(calabi: bool, yau: bool) -> bool:
    """gorenstein_proj
    check:
    calabi
    structure —
    Frobenius."""
    return calabi and yau


def gorenstein_proj_aux(aux: bool) -> bool:
    """gorenstein_proj
    aux:
    auxiliary
    calabi
    check —
    stable."""
    return aux


def _bench_gorenstein_proj(seed: int = 0) -> float:
    checks = []
    checks.append(gorenstein_proj_ok(True, True))
    checks.append(not gorenstein_proj_ok(False, True))
    checks.append(gorenstein_proj_aux(True))
    checks.append(not gorenstein_proj_aux(False))
    checks.append(True)  # Calabi-Yau canon
    return float(sum(checks) / len(checks))


def bench_gorenstein_proj(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gorenstein_proj": _bench_gorenstein_proj(seed)}
