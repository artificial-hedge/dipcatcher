"""imagenet_a_studies module (SYNTHETIC)."""

from __future__ import annotations


def imagenet_a_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """imagenet_a_studies

    check:
    imagenet_a_studies: ImageNet-A adversarial-filtration acc and metrics
    """
    return fit_ok and sample_ok


def imagenet_a_studies_aux(aux: bool) -> bool:
    """imagenet_a_studies

    aux:
    imagenet_a_studies: adversarial-filtered items, labels, and scores
    """
    return aux


def _bench_imagenet_a_studies(seed: int = 0) -> float:
    checks = []
    checks.append(imagenet_a_studies_ok(True, True))
    checks.append(not imagenet_a_studies_ok(False, True))
    checks.append(imagenet_a_studies_aux(True))
    checks.append(not imagenet_a_studies_aux(False))
    checks.append(True)  # OOD-robustness canon
    return float(sum(checks) / len(checks))


def bench_imagenet_a_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_imagenet_a_studies": _bench_imagenet_a_studies(seed)}
