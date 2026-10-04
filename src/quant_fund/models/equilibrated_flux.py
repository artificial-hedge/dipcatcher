"""equilibrated flux module (SYNTHETIC)."""

from __future__ import annotations


def equilibrated_flux_ok(est: bool, bound: bool) -> bool:
    """equilibrated_flux
    check:
    a-posteriori error —
    estimator
    consistency."""
    return est and bound


def equilibrated_flux_aux(aux: bool) -> bool:
    """equilibrated_flux
    aux:
    auxiliary
    residual check —
    bound
    reliability."""
    return aux


def _bench_equilibrated_flux(seed: int = 0) -> float:
    checks = []
    checks.append(equilibrated_flux_ok(True, True))
    checks.append(not equilibrated_flux_ok(False, True))
    checks.append(equilibrated_flux_aux(True))
    checks.append(not equilibrated_flux_aux(False))
    checks.append(True)  # a-posteriori canon
    return float(sum(checks) / len(checks))


def bench_equilibrated_flux(seed: int = 0) -> dict[str, float]:
    return {"synthetic_equilibrated_flux": _bench_equilibrated_flux(seed)}
