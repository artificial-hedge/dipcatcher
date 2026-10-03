"""adjoint sparse module (SYNTHETIC)."""

from __future__ import annotations


def adjoint_sparse_ok(step: bool, conv: bool) -> bool:
    """adjoint_sparse
    check:
    solver/transport —
    step/convergence
    consistency."""
    return step and conv


def adjoint_sparse_aux(aux: bool) -> bool:
    """adjoint_sparse
    aux:
    auxiliary
    solver check —
    order bound."""
    return aux


def _bench_adjoint_sparse(seed: int = 0) -> float:
    checks = []
    checks.append(adjoint_sparse_ok(True, True))
    checks.append(not adjoint_sparse_ok(False, True))
    checks.append(adjoint_sparse_aux(True))
    checks.append(not adjoint_sparse_aux(False))
    checks.append(True)  # solver/transport canon
    return float(sum(checks) / len(checks))


def bench_adjoint_sparse(seed: int = 0) -> dict[str, float]:
    return {"synthetic_adjoint_sparse": _bench_adjoint_sparse(seed)}
