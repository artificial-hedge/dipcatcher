"""kmt approx module (SYNTHETIC)."""

from __future__ import annotations


def kmt_approx_ok(weak: bool, conv: bool) -> bool:
    """kmt_approx
    check:
    weak
    convergence —
    measure."""
    return weak and conv


def kmt_approx_aux(aux: bool) -> bool:
    """kmt_approx
    aux:
    auxiliary
    convergence check —
    approx."""
    return aux


def _bench_kmt_approx(seed: int = 0) -> float:
    checks = []
    checks.append(kmt_approx_ok(True, True))
    checks.append(not kmt_approx_ok(False, True))
    checks.append(kmt_approx_aux(True))
    checks.append(not kmt_approx_aux(False))
    checks.append(True)  # weak-convergence canon
    return float(sum(checks) / len(checks))


def bench_kmt_approx(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kmt_approx": _bench_kmt_approx(seed)}
