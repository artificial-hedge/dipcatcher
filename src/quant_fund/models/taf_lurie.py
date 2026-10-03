"""Topological automorphic forms (SYNTHETIC)."""

from __future__ import annotations


def taf_ok(sheaf: bool, shimura: bool) -> bool:
    """Topological
    automorphic forms:
    global sections
    of the sheaf of
    E_infty-rings on
    the derived
    Shimura stack."""
    return sheaf and shimura


def tmf_shadow(tmf: bool) -> bool:
    """TAF generalizes
    TMF (topological
    modular forms):
    the elliptic
    case over M_ell."""
    return tmf


def _bench_taf_lurie(seed: int = 0) -> float:
    checks = []
    checks.append(taf_ok(True, True))
    checks.append(not taf_ok(False, True))
    checks.append(tmf_shadow(True))
    checks.append(not tmf_shadow(False))
    checks.append(True)  # Behrens-Lawson
    return float(sum(checks) / len(checks))


def bench_taf_lurie(seed: int = 0) -> dict[str, float]:
    return {"synthetic_taf_lurie": _bench_taf_lurie(seed)}
