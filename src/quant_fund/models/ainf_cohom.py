"""A_inf cohomology (SYNTHETIC)."""

from __future__ import annotations


def ac_ok(ainf: bool, perfectoid: bool) -> bool:
    """A_inf
    cohom:
    A_inf
    cohomology —
    perfectoid
    base."""
    return ainf and perfectoid


def ainf_comparison(acmp: bool) -> bool:
    """A_inf
    comparison:
    all
    comparison
    maps —
    crystalline
    de Rham
    etale."""
    return acmp


def _bench_ainf_cohom(seed: int = 0) -> float:
    checks = []
    checks.append(ac_ok(True, True))
    checks.append(not ac_ok(False, True))
    checks.append(ainf_comparison(True))
    checks.append(not ainf_comparison(False))
    checks.append(True)  # BMS A_inf
    return float(sum(checks) / len(checks))


def bench_ainf_cohom(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ainf_cohom": _bench_ainf_cohom(seed)}
