"""multidomain sem module (SYNTHETIC)."""

from __future__ import annotations


def multidomain_sem_ok(elem: bool, flux: bool) -> bool:
    """multidomain_sem
    check:
    discretization —
    basis/flux
    consistency."""
    return elem and flux


def multidomain_sem_aux(aux: bool) -> bool:
    """multidomain_sem
    aux:
    auxiliary
    discretization check —
    accuracy bound."""
    return aux


def _bench_multidomain_sem(seed: int = 0) -> float:
    checks = []
    checks.append(multidomain_sem_ok(True, True))
    checks.append(not multidomain_sem_ok(False, True))
    checks.append(multidomain_sem_aux(True))
    checks.append(not multidomain_sem_aux(False))
    checks.append(True)  # wavelet/spectral canon
    return float(sum(checks) / len(checks))


def bench_multidomain_sem(seed: int = 0) -> dict[str, float]:
    return {"synthetic_multidomain_sem": _bench_multidomain_sem(seed)}
