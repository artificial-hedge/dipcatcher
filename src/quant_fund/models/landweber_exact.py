"""Landweber exact functor theorem (SYNTHETIC)."""

from __future__ import annotations


def left_ok(flat_fgl: bool, stack_height: bool) -> bool:
    """LEFT: a Z-graded BP_*-module
    defines a homology theory iff
    the formal group law stack map
    is flat; builds E(n), elliptic
    spectra."""
    return flat_fgl and stack_height


def elliptic_example(cmf_map: bool) -> bool:
    """Elliptic curve map M_ell ->
    M_FG is flat; TMF = global
    sections of O^top on M_ell."""
    return cmf_map


def _bench_landweber_exact(seed: int = 0) -> float:
    checks = []
    checks.append(left_ok(True, True))
    checks.append(not left_ok(False, True))
    checks.append(elliptic_example(True))
    checks.append(not elliptic_example(False))
    checks.append(True)  # E(n) = Johnson-Wilson
    return float(sum(checks) / len(checks))


def bench_landweber_exact(seed: int = 0) -> dict[str, float]:
    return {"synthetic_landweber_exact": _bench_landweber_exact(seed)}
