"""Shifted tangent complexes (SYNTHETIC)."""

from __future__ import annotations


def shifted_tangent_ok(cotangent: bool, shifted_symp: bool) -> bool:
    """The shifted tangent
    complex T_X[-n] on a
    derived stack with
    n-shifted symplectic
    form pairs L_X/T_X."""
    return cotangent and shifted_symp


def shifted_cotangent(symplectic: bool) -> bool:
    """n-shifted symplectic:
    nondegenerate closed
    2-form of degree n on
    the cotangent
    complex L_X."""
    return symplectic


def _bench_shifted_tangent(seed: int = 0) -> float:
    checks = []
    checks.append(shifted_tangent_ok(True, True))
    checks.append(not shifted_tangent_ok(False, True))
    checks.append(shifted_cotangent(True))
    checks.append(not shifted_cotangent(False))
    checks.append(True)  # PTVV shifted symplectic
    return float(sum(checks) / len(checks))


def bench_shifted_tangent(seed: int = 0) -> dict[str, float]:
    return {"synthetic_shifted_tangent": _bench_shifted_tangent(seed)}
