"""imagenet_o_studies module (SYNTHETIC)."""

from __future__ import annotations


def imagenet_o_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """imagenet_o_studies

    check:
    imagenet_o_studies: ImageNet-O out-of-distribution detection and rates
    """
    return fit_ok and sample_ok


def imagenet_o_studies_aux(aux: bool) -> bool:
    """imagenet_o_studies

    aux:
    imagenet_o_studies: OOD scores, AUCs, and rejection rates
    """
    return aux


def _bench_imagenet_o_studies(seed: int = 0) -> float:
    checks = []
    checks.append(imagenet_o_studies_ok(True, True))
    checks.append(not imagenet_o_studies_ok(False, True))
    checks.append(imagenet_o_studies_aux(True))
    checks.append(not imagenet_o_studies_aux(False))
    checks.append(True)  # OOD-robustness canon
    return float(sum(checks) / len(checks))


def bench_imagenet_o_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_imagenet_o_studies": _bench_imagenet_o_studies(seed)}
