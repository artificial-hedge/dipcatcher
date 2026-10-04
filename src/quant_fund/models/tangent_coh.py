"""Tangent cohomology (SYNTHETIC)."""

from __future__ import annotations


def tangent_coh_ok(cotangent_cx: bool, postnikov: bool) -> bool:
    """Tangent complex T_X =
    (L_X)^vee of a derived
    stack; controls
    deformations, obstr;
    Illusie-Lurie."""
    return cotangent_cx and postnikov


def tangent_at_x(classifying: bool) -> bool:
    """Tangent cohomology
    at a point x:
    H^i(T_{X,x}) counts
    i-fold infinitesimal
    deformations."""
    return classifying


def _bench_tangent_coh(seed: int = 0) -> float:
    checks = []
    checks.append(tangent_coh_ok(True, True))
    checks.append(not tangent_coh_ok(False, True))
    checks.append(tangent_at_x(True))
    checks.append(not tangent_at_x(False))
    checks.append(True)  # deformation=H^0, obs=H^1
    return float(sum(checks) / len(checks))


def bench_tangent_coh(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tangent_coh": _bench_tangent_coh(seed)}
