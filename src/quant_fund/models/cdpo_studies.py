"""cdpo_studies module (SYNTHETIC)."""

from __future__ import annotations


def cdpo_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """cdpo_studies

    check:
    cdpo_studies: contrastive preference and gradient blending/margins and pairs
    """
    return fit_ok and sample_ok


def cdpo_studies_aux(aux: bool) -> bool:
    """cdpo_studies

    aux:
    cdpo_studies: negative gradients and stability/lens and updates
    """
    return aux


def _bench_cdpo_studies(seed: int = 0) -> float:
    checks = []
    checks.append(cdpo_studies_ok(True, True))
    checks.append(not cdpo_studies_ok(False, True))
    checks.append(cdpo_studies_aux(True))
    checks.append(not cdpo_studies_aux(False))
    checks.append(True)  # post-training-2 canon
    return float(sum(checks) / len(checks))


def bench_cdpo_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cdpo_studies": _bench_cdpo_studies(seed)}
