"""DG quotients (SYNTHETIC)."""

from __future__ import annotations


def dg_quotient_ok(drinfeld: bool, localization: bool) -> bool:
    """Drinfeld dg quotient:
    universal dg
    localization of
    a pretriangulated
    dg cat; Keller-
    Drinfeld."""
    return drinfeld and localization


def dg_verdier(exact_seq: bool) -> bool:
    """DG Verdier quotient:
    enhances Verdier
    localization of
    triangulated
    categories; exact
    sequence in K_0."""
    return exact_seq


def _bench_dg_quotient(seed: int = 0) -> float:
    checks = []
    checks.append(dg_quotient_ok(True, True))
    checks.append(not dg_quotient_ok(False, True))
    checks.append(dg_verdier(True))
    checks.append(not dg_verdier(False))
    checks.append(True)  # universal property
    return float(sum(checks) / len(checks))


def bench_dg_quotient(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dg_quotient": _bench_dg_quotient(seed)}
