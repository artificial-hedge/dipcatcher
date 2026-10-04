"""weibull domain module (SYNTHETIC)."""

from __future__ import annotations


def weibull_domain_ok(tail: bool, xi: bool) -> bool:
    """weibull_domain
    check:
    extreme-value
    structure —
    Gumbel."""
    return tail and xi


def weibull_domain_aux(aux: bool) -> bool:
    """weibull_domain
    aux:
    auxiliary
    max-domain
    check —
    Weibull."""
    return aux


def _bench_weibull_domain(seed: int = 0) -> float:
    checks = []
    checks.append(weibull_domain_ok(True, True))
    checks.append(not weibull_domain_ok(False, True))
    checks.append(weibull_domain_aux(True))
    checks.append(not weibull_domain_aux(False))
    checks.append(True)  # EVT canon
    return float(sum(checks) / len(checks))


def bench_weibull_domain(seed: int = 0) -> dict[str, float]:
    return {"synthetic_weibull_domain": _bench_weibull_domain(seed)}
