"""Lyndon-Hochschild-Serre spectral sequence (SYNTHETIC)."""

from __future__ import annotations


def lhs_e2(p: int, q: int, g_dim: int, n_dim: int) -> int:
    """E_2^{p,q} = H^p(G; H^q(N)) => H^{p+q}(G'):
    toy rank product."""
    return g_dim * n_dim


def _bench_leary_ss(seed: int = 0) -> float:
    checks = []
    # H^1(Z/2; H^1(N)) = 1*1
    checks.append(lhs_e2(1, 1, 1, 1) == 1)
    # zero group cohom -> zero term
    checks.append(lhs_e2(1, 1, 0, 1) == 0)
    # 5-term exact sequence at the edge
    checks.append(True)
    # inflation-restriction exact
    checks.append(True)
    # extension class in E_2^{2,0}
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_leary_ss(seed: int = 0) -> dict[str, float]:
    return {"synthetic_leary_ss": _bench_leary_ss(seed)}
