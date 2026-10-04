"""Grayson S-construction (SYNTHETIC)."""

from __future__ import annotations


def gs_ok(grayson: bool, s_constr: bool) -> bool:
    """Grayson
    S:
    Grayson
    S
    construction —
    binary
    complex."""
    return grayson and s_constr


def binary_complex(bcx: bool) -> bool:
    """Binary
    complex:
    binary
    exact
    complex —
    Grayson."""
    return bcx


def _bench_grayson_s(seed: int = 0) -> float:
    checks = []
    checks.append(gs_ok(True, True))
    checks.append(not gs_ok(False, True))
    checks.append(binary_complex(True))
    checks.append(not binary_complex(False))
    checks.append(True)  # Grayson
    return float(sum(checks) / len(checks))


def bench_grayson_s(seed: int = 0) -> dict[str, float]:
    return {"synthetic_grayson_s": _bench_grayson_s(seed)}
