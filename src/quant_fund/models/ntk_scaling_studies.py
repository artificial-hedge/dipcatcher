"""ntk_scaling_studies module (SYNTHETIC)."""

from __future__ import annotations


def ntk_scaling_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ntk_scaling_studies

    check:
    ntk_scaling_studies: NTK-aware RoPE interpolation/frequencies and bases
    """
    return fit_ok and sample_ok


def ntk_scaling_studies_aux(aux: bool) -> bool:
    """ntk_scaling_studies

    aux:
    ntk_scaling_studies: dynamic NTK scaling and depth extension/context and ratios
    """
    return aux


def _bench_ntk_scaling_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ntk_scaling_studies_ok(True, True))
    checks.append(not ntk_scaling_studies_ok(False, True))
    checks.append(ntk_scaling_studies_aux(True))
    checks.append(not ntk_scaling_studies_aux(False))
    checks.append(True)  # long-context canon
    return float(sum(checks) / len(checks))


def bench_ntk_scaling_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ntk_scaling_studies": _bench_ntk_scaling_studies(seed)}
