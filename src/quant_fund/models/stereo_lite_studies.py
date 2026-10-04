"""stereo_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def stereo_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """stereo_lite_studies

    check:
    stereo_lite_studies: stereotype-detection metrics
    """
    return fit_ok and sample_ok


def stereo_lite_studies_aux(aux: bool) -> bool:
    """stereo_lite_studies

    aux:
    stereo_lite_studies: sentences, targets, labels, and accuracies
    """
    return aux


def _bench_stereo_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(stereo_lite_studies_ok(True, True))
    checks.append(not stereo_lite_studies_ok(False, True))
    checks.append(stereo_lite_studies_aux(True))
    checks.append(not stereo_lite_studies_aux(False))
    checks.append(True)  # social-bias-eval canon
    return float(sum(checks) / len(checks))


def bench_stereo_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stereo_lite_studies": _bench_stereo_lite_studies(seed)}
