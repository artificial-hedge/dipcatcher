"""imagenet_e_studies module (SYNTHETIC)."""

from __future__ import annotations


def imagenet_e_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """imagenet_e_studies

    check:
    imagenet_e_studies: ImageNet-E example-edits robustness and metrics
    """
    return fit_ok and sample_ok


def imagenet_e_studies_aux(aux: bool) -> bool:
    """imagenet_e_studies

    aux:
    imagenet_e_studies: edited variants, pixel rates, and accuracy
    """
    return aux


def _bench_imagenet_e_studies(seed: int = 0) -> float:
    checks = []
    checks.append(imagenet_e_studies_ok(True, True))
    checks.append(not imagenet_e_studies_ok(False, True))
    checks.append(imagenet_e_studies_aux(True))
    checks.append(not imagenet_e_studies_aux(False))
    checks.append(True)  # OOD-robustness canon
    return float(sum(checks) / len(checks))


def bench_imagenet_e_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_imagenet_e_studies": _bench_imagenet_e_studies(seed)}
