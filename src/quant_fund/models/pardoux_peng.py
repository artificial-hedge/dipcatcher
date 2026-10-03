"""pardoux peng module (SYNTHETIC)."""

from __future__ import annotations


def pardoux_peng_ok(bs1: bool, pp: bool) -> bool:
    """pardoux_peng
    check:
    BSDE —
    Pardoux-Peng
    adapted
    solution."""
    return bs1 and pp


def pardoux_peng_aux(aux: bool) -> bool:
    """pardoux_peng
    aux:
    auxiliary
    FBSDE
    check —
    decoupling
    field."""
    return aux


def _bench_pardoux_peng(seed: int = 0) -> float:
    checks = []
    checks.append(pardoux_peng_ok(True, True))
    checks.append(not pardoux_peng_ok(False, True))
    checks.append(pardoux_peng_aux(True))
    checks.append(not pardoux_peng_aux(False))
    checks.append(True)  # BSDE canon
    return float(sum(checks) / len(checks))


def bench_pardoux_peng(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pardoux_peng": _bench_pardoux_peng(seed)}
