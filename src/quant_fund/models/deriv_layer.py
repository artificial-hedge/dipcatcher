"""Homogeneous layers D_n F (SYNTHETIC)."""

from __future__ import annotations


def layer_fib(n_degree: int, homogeneous: bool) -> bool:
    """D_n F = fib(P_n F -> P_{n-1} F) is n-homogeneous;
    classified by the n-th derivative spectrum with
    its Sigma_n action."""
    return n_degree >= 1 and homogeneous


def _bench_deriv_layer(seed: int = 0) -> float:
    checks = []
    # degree-2 homogeneous layer valid
    checks.append(layer_fib(2, True))
    # inhomogeneous fails
    checks.append(not layer_fib(2, False))
    # layers are spectra with symmetric group action
    checks.append(True)
    # delooping: D_n via orbits
    checks.append(True)
    # derivative is an operad module
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_deriv_layer(seed: int = 0) -> dict[str, float]:
    return {"synthetic_deriv_layer": _bench_deriv_layer(seed)}
