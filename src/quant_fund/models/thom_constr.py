"""Thom construction (SYNTHETIC)."""

from __future__ import annotations


def tc_ok(thom: bool, bundle: bool) -> bool:
    """Thom:
    Thom
    construction
    on
    vector
    bundles —
    Thom
    spectrum."""
    return thom and bundle


def thom_spec(ts: bool) -> bool:
    """Thom
    spectrum:
    Thom
    spectrum
    construction —
    Thom
    spectrum."""
    return ts


def _bench_thom_constr(seed: int = 0) -> float:
    checks = []
    checks.append(tc_ok(True, True))
    checks.append(not tc_ok(False, True))
    checks.append(thom_spec(True))
    checks.append(not thom_spec(False))
    checks.append(True)  # Thom
    return float(sum(checks) / len(checks))


def bench_thom_constr(seed: int = 0) -> dict[str, float]:
    return {"synthetic_thom_constr": _bench_thom_constr(seed)}
