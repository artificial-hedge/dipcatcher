"""latent_reasoning_studies module (SYNTHETIC)."""

from __future__ import annotations


def latent_reasoning_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """latent_reasoning_studies

    check:
    latent_reasoning_studies: hidden-space computation and recurrence/compressed and internal
    """
    return fit_ok and sample_ok


def latent_reasoning_studies_aux(aux: bool) -> bool:
    """latent_reasoning_studies

    aux:
    latent_reasoning_studies: continuous thoughts and pause tokens/deep and silent
    """
    return aux


def _bench_latent_reasoning_studies(seed: int = 0) -> float:
    checks = []
    checks.append(latent_reasoning_studies_ok(True, True))
    checks.append(not latent_reasoning_studies_ok(False, True))
    checks.append(latent_reasoning_studies_aux(True))
    checks.append(not latent_reasoning_studies_aux(False))
    checks.append(True)  # inference-scaling canon
    return float(sum(checks) / len(checks))


def bench_latent_reasoning_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_latent_reasoning_studies": _bench_latent_reasoning_studies(seed)}
