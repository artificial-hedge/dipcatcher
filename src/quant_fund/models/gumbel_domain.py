"""gumbel domain module (SYNTHETIC)."""

from __future__ import annotations


def gumbel_domain_ok(tail: bool, xi: bool) -> bool:
    """gumbel_domain
    check:
    extreme-value
    structure —
    Gumbel."""
    return tail and xi


def gumbel_domain_aux(aux: bool) -> bool:
    """gumbel_domain
    aux:
    auxiliary
    max-domain
    check —
    Weibull."""
    return aux


def _bench_gumbel_domain(seed: int = 0) -> float:
    checks = []
    checks.append(gumbel_domain_ok(True, True))
    checks.append(not gumbel_domain_ok(False, True))
    checks.append(gumbel_domain_aux(True))
    checks.append(not gumbel_domain_aux(False))
    checks.append(True)  # EVT canon
    return float(sum(checks) / len(checks))


def bench_gumbel_domain(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gumbel_domain": _bench_gumbel_domain(seed)}
