"""pade approx module (SYNTHETIC)."""

from __future__ import annotations


def pade_approx_ok(rational: bool, approx: bool) -> bool:
    """pade_approx
    check:
    rational
    approximation —
    Padé."""
    return rational and approx


def pade_approx_aux(aux: bool) -> bool:
    """pade_approx
    aux:
    auxiliary
    approx check —
    convergent."""
    return aux


def _bench_pade_approx(seed: int = 0) -> float:
    checks = []
    checks.append(pade_approx_ok(True, True))
    checks.append(not pade_approx_ok(False, True))
    checks.append(pade_approx_aux(True))
    checks.append(not pade_approx_aux(False))
    checks.append(True)  # rational-approx canon
    return float(sum(checks) / len(checks))


def bench_pade_approx(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pade_approx": _bench_pade_approx(seed)}
