"""epi rk module (SYNTHETIC)."""

from __future__ import annotations


def epi_rk_ok(step: bool, conv: bool) -> bool:
    """epi_rk
    check:
    solver/transport —
    step/convergence
    consistency."""
    return step and conv


def epi_rk_aux(aux: bool) -> bool:
    """epi_rk
    aux:
    auxiliary
    solver check —
    order bound."""
    return aux


def _bench_epi_rk(seed: int = 0) -> float:
    checks = []
    checks.append(epi_rk_ok(True, True))
    checks.append(not epi_rk_ok(False, True))
    checks.append(epi_rk_aux(True))
    checks.append(not epi_rk_aux(False))
    checks.append(True)  # solver/transport canon
    return float(sum(checks) / len(checks))


def bench_epi_rk(seed: int = 0) -> dict[str, float]:
    return {"synthetic_epi_rk": _bench_epi_rk(seed)}
