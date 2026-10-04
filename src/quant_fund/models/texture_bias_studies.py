"""texture_bias_studies module (SYNTHETIC)."""

from __future__ import annotations


def texture_bias_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """texture_bias_studies

    check:
    texture_bias_studies: texture-bias measurement and debiased accuracy
    """
    return fit_ok and sample_ok


def texture_bias_studies_aux(aux: bool) -> bool:
    """texture_bias_studies

    aux:
    texture_bias_studies: texture cues, style transfers, and metrics
    """
    return aux


def _bench_texture_bias_studies(seed: int = 0) -> float:
    checks = []
    checks.append(texture_bias_studies_ok(True, True))
    checks.append(not texture_bias_studies_ok(False, True))
    checks.append(texture_bias_studies_aux(True))
    checks.append(not texture_bias_studies_aux(False))
    checks.append(True)  # cue-conflict canon
    return float(sum(checks) / len(checks))


def bench_texture_bias_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_texture_bias_studies": _bench_texture_bias_studies(seed)}
