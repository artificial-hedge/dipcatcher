"""Krull dimension of finite-varieties via prime-ideal chains (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.spec_ring import is_prime_ideal_zn, spec_zn


def krull_dim_zn(n: int) -> int:
    """Longest chain of prime ideals in Z/nZ."""
    sp = spec_zn(n)
    best = 0
    for a in sp:
        for b in sp:
            if a < b:  # strict containment
                chain_len = 1
                for c in sp:
                    if b < c:
                        chain_len = 2
                best = max(best, chain_len)
    return best


def is_variety_finite(n_pts: int) -> int:
    """Finite point set => dimension 0."""
    return 0


def _bench_variety_dim(seed: int = 0) -> float:
    checks = []
    # Z/4 = local artinian: single prime (2), dim 0
    checks.append(krull_dim_zn(4) == 0)
    # Z/6 = product of fields: all primes maximal, dim 0
    checks.append(krull_dim_zn(6) == 0)
    # Z/8: single prime (2), dim 0
    checks.append(krull_dim_zn(8) == 0)
    # finite rings are artinian => always dim 0; sanity on Z/12
    checks.append(krull_dim_zn(12) == 0)
    checks.append(is_variety_finite(3) == 0)
    checks.append(is_prime_ideal_zn(6, frozenset({0, 2, 4})))
    return float(sum(checks) / len(checks))


def bench_variety_dim(seed: int = 0) -> dict[str, float]:
    return {"synthetic_variety_dim": _bench_variety_dim(seed)}
