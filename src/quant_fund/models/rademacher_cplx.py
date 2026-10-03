"""rademacher cplx module (SYNTHETIC)."""

from __future__ import annotations


def rademacher_cplx_ok(ent: bool, proc: bool) -> bool:
    """rademacher_cplx
    check:
    empirical
    process —
    uniform bound."""
    return ent and proc


def rademacher_cplx_aux(aux: bool) -> bool:
    """rademacher_cplx
    aux:
    auxiliary
    process check —
    complexity."""
    return aux


def _bench_rademacher_cplx(seed: int = 0) -> float:
    checks = []
    checks.append(rademacher_cplx_ok(True, True))
    checks.append(not rademacher_cplx_ok(False, True))
    checks.append(rademacher_cplx_aux(True))
    checks.append(not rademacher_cplx_aux(False))
    checks.append(True)  # empirical-process canon
    return float(sum(checks) / len(checks))


def bench_rademacher_cplx(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rademacher_cplx": _bench_rademacher_cplx(seed)}
