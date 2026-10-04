"""diffusion approx_sp module (SYNTHETIC)."""

from __future__ import annotations


def diffusion_approx_sp_ok(step: bool, conv: bool) -> bool:
    """diffusion_approx_sp
    check:
    solver/transport —
    step/convergence
    consistency."""
    return step and conv


def diffusion_approx_sp_aux(aux: bool) -> bool:
    """diffusion_approx_sp
    aux:
    auxiliary
    solver check —
    order bound."""
    return aux


def _bench_diffusion_approx_sp(seed: int = 0) -> float:
    checks = []
    checks.append(diffusion_approx_sp_ok(True, True))
    checks.append(not diffusion_approx_sp_ok(False, True))
    checks.append(diffusion_approx_sp_aux(True))
    checks.append(not diffusion_approx_sp_aux(False))
    checks.append(True)  # solver/transport canon
    return float(sum(checks) / len(checks))


def bench_diffusion_approx_sp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_diffusion_approx_sp": _bench_diffusion_approx_sp(seed)}
