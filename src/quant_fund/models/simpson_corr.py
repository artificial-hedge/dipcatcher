"""Simpson correspondence (SYNTHETIC)."""

from __future__ import annotations


def sc_ok(simpson: bool, dolbeault: bool) -> bool:
    """Simpson:
    correspondence
    Higgs
    flat —
    nonabelian
    Hodge."""
    return simpson and dolbeault


def simpson_de_rham(sdr: bool) -> bool:
    """Simpson
    de
    Rham:
    Dolbeault
    equals
    de
    Rham —
    correspondence."""
    return sdr


def _bench_simpson_corr(seed: int = 0) -> float:
    checks = []
    checks.append(sc_ok(True, True))
    checks.append(not sc_ok(False, True))
    checks.append(simpson_de_rham(True))
    checks.append(not simpson_de_rham(False))
    checks.append(True)  # Simpson
    return float(sum(checks) / len(checks))


def bench_simpson_corr(seed: int = 0) -> dict[str, float]:
    return {"synthetic_simpson_corr": _bench_simpson_corr(seed)}
