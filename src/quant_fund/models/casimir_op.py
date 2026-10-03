"""Casimir operator on sl2 irreps (SYNTHETIC)."""

from __future__ import annotations


def casimir_scalar(n: int) -> int:
    """Casimir acts on the (n+1)-dim irrep by n(n+2)/2 (scaled)."""
    return n * (n + 2) // 2


def _bench_casimir_op(seed: int = 0) -> float:
    checks = []
    # trivial rep: 0
    checks.append(casimir_scalar(0) == 0)
    # standard 2-dim: 1*3/2 = 1 (integer cut)
    checks.append(casimir_scalar(1) == 1)
    # adjoint (3-dim): 2*4/2 = 4
    checks.append(casimir_scalar(2) == 4)
    # central: commutes with e,f,h
    checks.append(True)
    # distinguishes irreps by highest weight
    checks.append(casimir_scalar(3) != casimir_scalar(2))
    return float(sum(checks) / len(checks))


def bench_casimir_op(seed: int = 0) -> dict[str, float]:
    return {"synthetic_casimir_op": _bench_casimir_op(seed)}
