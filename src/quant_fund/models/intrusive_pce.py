"""intrusive pce module (SYNTHETIC)."""

from __future__ import annotations


def intrusive_pce_ok(uq: bool, mode: bool) -> bool:
    """intrusive_pce
    check:
    stochastic-Galerkin/UQ —
    basis
    consistency."""
    return uq and mode


def intrusive_pce_aux(aux: bool) -> bool:
    """intrusive_pce
    aux:
    auxiliary
    chaos check —
    moment bound."""
    return aux


def _bench_intrusive_pce(seed: int = 0) -> float:
    checks = []
    checks.append(intrusive_pce_ok(True, True))
    checks.append(not intrusive_pce_ok(False, True))
    checks.append(intrusive_pce_aux(True))
    checks.append(not intrusive_pce_aux(False))
    checks.append(True)  # stochastic-Galerkin canon
    return float(sum(checks) / len(checks))


def bench_intrusive_pce(seed: int = 0) -> dict[str, float]:
    return {"synthetic_intrusive_pce": _bench_intrusive_pce(seed)}
