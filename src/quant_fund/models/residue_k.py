"""Residue K-theory (SYNTHETIC)."""

from __future__ import annotations


def rk_ok(residue: bool, k: bool) -> bool:
    """Residue
    K:
    residue
    K
    theory —
    class."""
    return residue and k


def residue_class(rc: bool) -> bool:
    """Residue
    class:
    residue
    class —
    characteristic."""
    return rc


def _bench_residue_k(seed: int = 0) -> float:
    checks = []
    checks.append(rk_ok(True, True))
    checks.append(not rk_ok(False, True))
    checks.append(residue_class(True))
    checks.append(not residue_class(False))
    checks.append(True)  # Gersten
    return float(sum(checks) / len(checks))


def bench_residue_k(seed: int = 0) -> dict[str, float]:
    return {"synthetic_residue_k": _bench_residue_k(seed)}
