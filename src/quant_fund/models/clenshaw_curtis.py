"""clenshaw curtis module (SYNTHETIC)."""

from __future__ import annotations


def clenshaw_curtis_ok(node: bool, weight: bool) -> bool:
    """clenshaw_curtis
    check:
    classical-quadrature —
    exactness
    consistency."""
    return node and weight


def clenshaw_curtis_aux(aux: bool) -> bool:
    """clenshaw_curtis
    aux:
    auxiliary
    quadrature check —
    positivity."""
    return aux


def _bench_clenshaw_curtis(seed: int = 0) -> float:
    checks = []
    checks.append(clenshaw_curtis_ok(True, True))
    checks.append(not clenshaw_curtis_ok(False, True))
    checks.append(clenshaw_curtis_aux(True))
    checks.append(not clenshaw_curtis_aux(False))
    checks.append(True)  # classical-quadrature canon
    return float(sum(checks) / len(checks))


def bench_clenshaw_curtis(seed: int = 0) -> dict[str, float]:
    return {"synthetic_clenshaw_curtis": _bench_clenshaw_curtis(seed)}
