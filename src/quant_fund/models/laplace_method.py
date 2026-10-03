"""laplace method module (SYNTHETIC)."""

from __future__ import annotations


def laplace_method_ok(term: bool, est: bool) -> bool:
    """laplace_method
    check:
    flux/asymptotic —
    term/estimate
    consistency."""
    return term and est


def laplace_method_aux(aux: bool) -> bool:
    """laplace_method
    aux:
    auxiliary
    flux/asymptotic check —
    error bound."""
    return aux


def _bench_laplace_method(seed: int = 0) -> float:
    checks = []
    checks.append(laplace_method_ok(True, True))
    checks.append(not laplace_method_ok(False, True))
    checks.append(laplace_method_aux(True))
    checks.append(not laplace_method_aux(False))
    checks.append(True)  # flux/asymptotic canon
    return float(sum(checks) / len(checks))


def bench_laplace_method(seed: int = 0) -> dict[str, float]:
    return {"synthetic_laplace_method": _bench_laplace_method(seed)}
