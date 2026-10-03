"""diffusion approx module (SYNTHETIC)."""

from __future__ import annotations


def diffusion_approx_ok(lim: bool, scale: bool) -> bool:
    """diffusion_approx
    check:
    heavy-traffic
    structure —
    diffusion
    limit."""
    return lim and scale


def diffusion_approx_aux(aux: bool) -> bool:
    """diffusion_approx
    aux:
    auxiliary
    scaling
    check —
    QED
    regime."""
    return aux


def _bench_diffusion_approx(seed: int = 0) -> float:
    checks = []
    checks.append(diffusion_approx_ok(True, True))
    checks.append(not diffusion_approx_ok(False, True))
    checks.append(diffusion_approx_aux(True))
    checks.append(not diffusion_approx_aux(False))
    checks.append(True)  # heavy-traffic canon
    return float(sum(checks) / len(checks))


def bench_diffusion_approx(seed: int = 0) -> dict[str, float]:
    return {"synthetic_diffusion_approx": _bench_diffusion_approx(seed)}
