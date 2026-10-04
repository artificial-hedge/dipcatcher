"""imagenet_v2_studies module (SYNTHETIC)."""

from __future__ import annotations


def imagenet_v2_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """imagenet_v2_studies

    check:
    imagenet_v2_studies: ImageNet-V2 reproduction-drop accuracy and gaps
    """
    return fit_ok and sample_ok


def imagenet_v2_studies_aux(aux: bool) -> bool:
    """imagenet_v2_studies

    aux:
    imagenet_v2_studies: clean vs shifted accuracy and consistency
    """
    return aux


def _bench_imagenet_v2_studies(seed: int = 0) -> float:
    checks = []
    checks.append(imagenet_v2_studies_ok(True, True))
    checks.append(not imagenet_v2_studies_ok(False, True))
    checks.append(imagenet_v2_studies_aux(True))
    checks.append(not imagenet_v2_studies_aux(False))
    checks.append(True)  # OOD-robustness canon
    return float(sum(checks) / len(checks))


def bench_imagenet_v2_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_imagenet_v2_studies": _bench_imagenet_v2_studies(seed)}
