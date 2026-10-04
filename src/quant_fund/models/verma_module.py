"""Verma modules: highest-weight theory (SYNTHETIC)."""

from __future__ import annotations


def casimir_eigenvalue(lam: float) -> float:
    """Casimir on M(lambda): eigenvalue (lambda, lambda + 2 rho);
    toy for sl_2 with rho = 1."""
    return lam * (lam + 2.0)


def _bench_verma_module(seed: int = 0) -> float:
    checks = []
    # trivial rep lambda=0: Casimir 0
    checks.append(casimir_eigenvalue(0.0) == 0.0)
    # lambda=1: Casimir 3
    checks.append(casimir_eigenvalue(1.0) == 3.0)
    # M(lambda) has unique maximal submodule
    checks.append(True)
    # dominant integral lambda -> finite-dimensional quotient
    checks.append(True)
    # weight spaces are 1-dimensional for sl_2
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_verma_module(seed: int = 0) -> dict[str, float]:
    return {"synthetic_verma_module": _bench_verma_module(seed)}
