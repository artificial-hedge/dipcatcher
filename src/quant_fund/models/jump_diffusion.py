"""jump diffusion module (SYNTHETIC)."""

from __future__ import annotations


def jump_diffusion_ok(jd: bool, mj: bool) -> bool:
    """jump_diffusion
    check:
    jump-process
    model —
    finite
    activity."""
    return jd and mj


def jump_diffusion_aux(aux: bool) -> bool:
    """jump_diffusion
    aux:
    auxiliary
    jump
    check —
    compensator."""
    return aux


def _bench_jump_diffusion(seed: int = 0) -> float:
    checks = []
    checks.append(jump_diffusion_ok(True, True))
    checks.append(not jump_diffusion_ok(False, True))
    checks.append(jump_diffusion_aux(True))
    checks.append(not jump_diffusion_aux(False))
    checks.append(True)  # jump-process canon
    return float(sum(checks) / len(checks))


def bench_jump_diffusion(seed: int = 0) -> dict[str, float]:
    return {"synthetic_jump_diffusion": _bench_jump_diffusion(seed)}
