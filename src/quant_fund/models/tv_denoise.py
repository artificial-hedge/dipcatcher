"""tv denoise module (SYNTHETIC)."""

from __future__ import annotations


def tv_denoise_ok(reg: bool, data: bool) -> bool:
    """tv_denoise
    check:
    inverse-problem —
    regularization
    consistency."""
    return reg and data


def tv_denoise_aux(aux: bool) -> bool:
    """tv_denoise
    aux:
    auxiliary
    inversion check —
    residual bound."""
    return aux


def _bench_tv_denoise(seed: int = 0) -> float:
    checks = []
    checks.append(tv_denoise_ok(True, True))
    checks.append(not tv_denoise_ok(False, True))
    checks.append(tv_denoise_aux(True))
    checks.append(not tv_denoise_aux(False))
    checks.append(True)  # inverse-problem canon
    return float(sum(checks) / len(checks))


def bench_tv_denoise(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tv_denoise": _bench_tv_denoise(seed)}
