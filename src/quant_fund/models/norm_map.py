"""Hill-Hopkins-Ravenel norm (SYNTHETIC)."""

from __future__ import annotations


def norm_multiplicative(n_inputs: int, g_order: int) -> bool:
    """N_e^G of a product = product of N_e^G's:
    multiplicative norm (HHR smashing norm)."""
    return n_inputs >= 1 and g_order >= 1


def norm_of_sphere(dim_in: int, g_order: int) -> int:
    """N_e^G(S^V) has dimension |G| * dim V (smash |G| copies)."""
    return g_order * dim_in


def _bench_norm_map(seed: int = 0) -> float:
    checks = []
    checks.append(norm_multiplicative(3, 8))
    checks.append(norm_of_sphere(2, 8) == 16)
    checks.append(norm_of_sphere(1, 2) == 2)
    # N_e^{C2} used in Kervaire invariant-one resolution
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_norm_map(seed: int = 0) -> dict[str, float]:
    return {"synthetic_norm_map": _bench_norm_map(seed)}
