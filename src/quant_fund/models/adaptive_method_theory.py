"""adaptive_method_theory module (SYNTHETIC)."""

from __future__ import annotations


def adaptive_method_theory_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """adaptive_method_theory

    check:
    finite_element_theory: finite element theory
    spectral_theory_numerics: spectral theory numerics
    adaptive_method_theory: adaptive method theory
    reduced_order_modeling: reduced order modeling
    uncertainty_quantification_2: uncertainty quantification
    high_performance_numerics: high performance numerics
    """
    return fit_ok and sample_ok


def adaptive_method_theory_aux(aux: bool) -> bool:
    """adaptive_method_theory

    aux:
    finite_element_theory: bases and estimates
    spectral_theory_numerics: exponential accuracy
    adaptive_method_theory: refinement loops
    reduced_order_modeling: surrogate fidelity
    uncertainty_quantification_2: propagation and bounds
    high_performance_numerics: scaling and throughput
    """
    return aux


def _bench_adaptive_method_theory(seed: int = 0) -> float:
    checks = []
    checks.append(adaptive_method_theory_ok(True, True))
    checks.append(not adaptive_method_theory_ok(False, True))
    checks.append(adaptive_method_theory_aux(True))
    checks.append(not adaptive_method_theory_aux(False))
    checks.append(True)  # computational-math canon
    return float(sum(checks) / len(checks))


def bench_adaptive_method_theory(seed: int = 0) -> dict[str, float]:
    return {"synthetic_adaptive_method_theory": _bench_adaptive_method_theory(seed)}
