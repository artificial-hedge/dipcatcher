"""imagenet_bg_studies module (SYNTHETIC)."""

from __future__ import annotations


def imagenet_bg_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """imagenet_bg_studies

    check:
    imagenet_bg_studies: ImageNet-BG background-swap classification and rates
    """
    return fit_ok and sample_ok


def imagenet_bg_studies_aux(aux: bool) -> bool:
    """imagenet_bg_studies

    aux:
    imagenet_bg_studies: background variants, robustness, and scores
    """
    return aux


def _bench_imagenet_bg_studies(seed: int = 0) -> float:
    checks = []
    checks.append(imagenet_bg_studies_ok(True, True))
    checks.append(not imagenet_bg_studies_ok(False, True))
    checks.append(imagenet_bg_studies_aux(True))
    checks.append(not imagenet_bg_studies_aux(False))
    checks.append(True)  # cue-conflict canon
    return float(sum(checks) / len(checks))


def bench_imagenet_bg_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_imagenet_bg_studies": _bench_imagenet_bg_studies(seed)}
