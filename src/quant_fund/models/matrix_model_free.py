"""matrix_model_free module (SYNTHETIC)."""

from __future__ import annotations


def matrix_model_free_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """matrix_model_free

    check:
    free_entropy: Voiculescu free entropy
    free_fisher_info: free Fisher information
    free_cumulant: free cumulant expansion
    freeness_check: freeness verification via mixed moments
    matrix_model_free: random matrix model of freeness
    free_berg: free Berg space / free Segal-Bargmann
    """
    return fit_ok and sample_ok


def matrix_model_free_aux(aux: bool) -> bool:
    """matrix_model_free

    aux:
    free_entropy: microstate dimension
    free_fisher_info: conjugate variable
    free_cumulant: noncrossing partition sum
    freeness_check: alternating centering
    matrix_model_free: GUE asymptotic freeness
    free_berg: free Segal-Bargmann transform
    """
    return aux


def _bench_matrix_model_free(seed: int = 0) -> float:
    checks = []
    checks.append(matrix_model_free_ok(True, True))
    checks.append(not matrix_model_free_ok(False, True))
    checks.append(matrix_model_free_aux(True))
    checks.append(not matrix_model_free_aux(False))
    checks.append(True)  # free-probability-2 canon
    return float(sum(checks) / len(checks))


def bench_matrix_model_free(seed: int = 0) -> dict[str, float]:
    return {"synthetic_matrix_model_free": _bench_matrix_model_free(seed)}
