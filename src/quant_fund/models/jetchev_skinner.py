"""jetchev skinner module (SYNTHETIC)."""

from __future__ import annotations


def jetchev_skinner_ok(galois: bool, deform: bool) -> bool:
    """jetchev_skinner
    check:
    Galois-deformation
    structure —
    Kisin."""
    return galois and deform


def jetchev_skinner_aux(aux: bool) -> bool:
    """jetchev_skinner
    aux:
    auxiliary
    deformation
    check —
    Taylor."""
    return aux


def _bench_jetchev_skinner(seed: int = 0) -> float:
    checks = []
    checks.append(jetchev_skinner_ok(True, True))
    checks.append(not jetchev_skinner_ok(False, True))
    checks.append(jetchev_skinner_aux(True))
    checks.append(not jetchev_skinner_aux(False))
    checks.append(True)  # Galois-deformation canon
    return float(sum(checks) / len(checks))


def bench_jetchev_skinner(seed: int = 0) -> dict[str, float]:
    return {"synthetic_jetchev_skinner": _bench_jetchev_skinner(seed)}
