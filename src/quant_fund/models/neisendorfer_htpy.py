"""Neisendorfer homotopy theory (SYNTHETIC)."""

from __future__ import annotations


def nh_ok(neisendorfer: bool, exponents: bool) -> bool:
    """Neisendorfer
    exponents:
    homotopy
    exponents —
    mod-p
    Moore."""
    return neisendorfer and exponents


def homotopy_exponents(he: bool) -> bool:
    """Homotopy
    exponents:
    p-primary
    exponents —
    spheres."""
    return he


def _bench_neisendorfer_htpy(seed: int = 0) -> float:
    checks = []
    checks.append(nh_ok(True, True))
    checks.append(not nh_ok(False, True))
    checks.append(homotopy_exponents(True))
    checks.append(not homotopy_exponents(False))
    checks.append(True)  # Neisendorfer
    return float(sum(checks) / len(checks))


def bench_neisendorfer_htpy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_neisendorfer_htpy": _bench_neisendorfer_htpy(seed)}
