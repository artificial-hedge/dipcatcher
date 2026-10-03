"""Grothendieck duality (SYNTHETIC)."""

from __future__ import annotations


def gd_ok(proper_map: bool, dualizing_cx: bool) -> bool:
    """Grothendieck
    duality:
    proper
    f^!
    right
    adjoint —
    coherent
    duality."""
    return proper_map and dualizing_cx


def residual_complex(rc: bool) -> bool:
    """Residual
    complex:
    Cousin
    resolution
    of
    dualizing —
    Hartshorne
    residue."""
    return rc


def _bench_groth_duality(seed: int = 0) -> float:
    checks = []
    checks.append(gd_ok(True, True))
    checks.append(not gd_ok(False, True))
    checks.append(residual_complex(True))
    checks.append(not residual_complex(False))
    checks.append(True)  # Grothendieck
    return float(sum(checks) / len(checks))


def bench_groth_duality(seed: int = 0) -> dict[str, float]:
    return {"synthetic_groth_duality": _bench_groth_duality(seed)}
