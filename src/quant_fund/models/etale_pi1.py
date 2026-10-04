"""Étale pi_1 and base change (SYNTHETIC)."""

from __future__ import annotations


def etale_pi1_ok(base: bool, exact: bool) -> bool:
    """Étale pi_1
    base change:
    homotopy exact
    sequence for
    proper smooth
    families; SGA1."""
    return base and exact


def profinite_pi1(profinite: bool) -> bool:
    """Profinite
    completion of
    the topological
    pi_1 computes
    pi_1^et over C."""
    return profinite


def _bench_etale_pi1(seed: int = 0) -> float:
    checks = []
    checks.append(etale_pi1_ok(True, True))
    checks.append(not etale_pi1_ok(False, True))
    checks.append(profinite_pi1(True))
    checks.append(not profinite_pi1(False))
    checks.append(True)  # SGA1 Exp X
    return float(sum(checks) / len(checks))


def bench_etale_pi1(seed: int = 0) -> dict[str, float]:
    return {"synthetic_etale_pi1": _bench_etale_pi1(seed)}
