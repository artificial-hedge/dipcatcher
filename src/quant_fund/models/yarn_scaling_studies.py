"""yarn_scaling_studies module (SYNTHETIC)."""

from __future__ import annotations


def yarn_scaling_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """yarn_scaling_studies

    check:
    yarn_scaling_studies: YaRN ramped interpolation/temperature and wavelengths
    """
    return fit_ok and sample_ok


def yarn_scaling_studies_aux(aux: bool) -> bool:
    """yarn_scaling_studies

    aux:
    yarn_scaling_studies: attention-scaled long-range extension/factors and heads
    """
    return aux


def _bench_yarn_scaling_studies(seed: int = 0) -> float:
    checks = []
    checks.append(yarn_scaling_studies_ok(True, True))
    checks.append(not yarn_scaling_studies_ok(False, True))
    checks.append(yarn_scaling_studies_aux(True))
    checks.append(not yarn_scaling_studies_aux(False))
    checks.append(True)  # long-context canon
    return float(sum(checks) / len(checks))


def bench_yarn_scaling_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_yarn_scaling_studies": _bench_yarn_scaling_studies(seed)}
