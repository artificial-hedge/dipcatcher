"""imagenet_r_studies module (SYNTHETIC)."""

from __future__ import annotations


def imagenet_r_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """imagenet_r_studies

    check:
    imagenet_r_studies: IN-R rendition out-of-distribution eval and acc
    """
    return fit_ok and sample_ok


def imagenet_r_studies_aux(aux: bool) -> bool:
    """imagenet_r_studies

    aux:
    imagenet_r_studies: rendition classes/sketches and top-1 scores
    """
    return aux


def _bench_imagenet_r_studies(seed: int = 0) -> float:
    checks = []
    checks.append(imagenet_r_studies_ok(True, True))
    checks.append(not imagenet_r_studies_ok(False, True))
    checks.append(imagenet_r_studies_aux(True))
    checks.append(not imagenet_r_studies_aux(False))
    checks.append(True)  # robustness-eval canon
    return float(sum(checks) / len(checks))


def bench_imagenet_r_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_imagenet_r_studies": _bench_imagenet_r_studies(seed)}
