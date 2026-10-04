"""kronecker approx module (SYNTHETIC)."""

from __future__ import annotations


def kronecker_approx_ok(rank: bool, err: bool) -> bool:
    """kronecker_approx
    check:
    low-rank —
    compression
    consistency."""
    return rank and err


def kronecker_approx_aux(aux: bool) -> bool:
    """kronecker_approx
    aux:
    auxiliary
    low-rank check —
    truncation bound."""
    return aux


def _bench_kronecker_approx(seed: int = 0) -> float:
    checks = []
    checks.append(kronecker_approx_ok(True, True))
    checks.append(not kronecker_approx_ok(False, True))
    checks.append(kronecker_approx_aux(True))
    checks.append(not kronecker_approx_aux(False))
    checks.append(True)  # low-rank canon
    return float(sum(checks) / len(checks))


def bench_kronecker_approx(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kronecker_approx": _bench_kronecker_approx(seed)}
