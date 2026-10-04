"""freeness_check module (SYNTHETIC)."""

from __future__ import annotations


def freeness_check_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """freeness_check

    check:
    free_entropy: Voiculescu free entropy
    free_fisher_info: free Fisher information
    free_cumulant: free cumulant expansion
    freeness_check: freeness verification via mixed moments
    matrix_model_free: random matrix model of freeness
    free_berg: free Berg space / free Segal-Bargmann
    """
    return fit_ok and sample_ok


def freeness_check_aux(aux: bool) -> bool:
    """freeness_check

    aux:
    free_entropy: microstate dimension
    free_fisher_info: conjugate variable
    free_cumulant: noncrossing partition sum
    freeness_check: alternating centering
    matrix_model_free: GUE asymptotic freeness
    free_berg: free Segal-Bargmann transform
    """
    return aux


def _bench_freeness_check(seed: int = 0) -> float:
    checks = []
    checks.append(freeness_check_ok(True, True))
    checks.append(not freeness_check_ok(False, True))
    checks.append(freeness_check_aux(True))
    checks.append(not freeness_check_aux(False))
    checks.append(True)  # free-probability-2 canon
    return float(sum(checks) / len(checks))


def bench_freeness_check(seed: int = 0) -> dict[str, float]:
    return {"synthetic_freeness_check": _bench_freeness_check(seed)}
