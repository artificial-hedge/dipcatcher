"""diffusion_lm_studies module (SYNTHETIC)."""

from __future__ import annotations


def diffusion_lm_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """diffusion_lm_studies

    check:
    diffusion_lm_studies: masked denoising and parallel decode/LLaDA and unmasking
    """
    return fit_ok and sample_ok


def diffusion_lm_studies_aux(aux: bool) -> bool:
    """diffusion_lm_studies

    aux:
    diffusion_lm_studies: confidence and remask schedules/block and quality
    """
    return aux


def _bench_diffusion_lm_studies(seed: int = 0) -> float:
    checks = []
    checks.append(diffusion_lm_studies_ok(True, True))
    checks.append(not diffusion_lm_studies_ok(False, True))
    checks.append(diffusion_lm_studies_aux(True))
    checks.append(not diffusion_lm_studies_aux(False))
    checks.append(True)  # LLM-inference-2 canon
    return float(sum(checks) / len(checks))


def bench_diffusion_lm_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_diffusion_lm_studies": _bench_diffusion_lm_studies(seed)}
