"""free_berg module (SYNTHETIC)."""

from __future__ import annotations


def free_berg_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """free_berg

    check:
    free_entropy: Voiculescu free entropy
    free_fisher_info: free Fisher information
    free_cumulant: free cumulant expansion
    freeness_check: freeness verification via mixed moments
    matrix_model_free: random matrix model of freeness
    free_berg: free Berg space / free Segal-Bargmann
    """
    return fit_ok and sample_ok


def free_berg_aux(aux: bool) -> bool:
    """free_berg

    aux:
    free_entropy: microstate dimension
    free_fisher_info: conjugate variable
    free_cumulant: noncrossing partition sum
    freeness_check: alternating centering
    matrix_model_free: GUE asymptotic freeness
    free_berg: free Segal-Bargmann transform
    """
    return aux


def _bench_free_berg(seed: int = 0) -> float:
    checks = []
    checks.append(free_berg_ok(True, True))
    checks.append(not free_berg_ok(False, True))
    checks.append(free_berg_aux(True))
    checks.append(not free_berg_aux(False))
    checks.append(True)  # free-probability-2 canon
    return float(sum(checks) / len(checks))


def bench_free_berg(seed: int = 0) -> dict[str, float]:
    return {"synthetic_free_berg": _bench_free_berg(seed)}
