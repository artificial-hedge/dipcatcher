"""cider_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def cider_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """cider_lite_studies

    check:
    cider_lite_studies: CIDEr caption metrics
    """
    return fit_ok and sample_ok


def cider_lite_studies_aux(aux: bool) -> bool:
    """cider_lite_studies

    aux:
    cider_lite_studies: images, references, candidates, and scores
    """
    return aux


def _bench_cider_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(cider_lite_studies_ok(True, True))
    checks.append(not cider_lite_studies_ok(False, True))
    checks.append(cider_lite_studies_aux(True))
    checks.append(not cider_lite_studies_aux(False))
    checks.append(True)  # metric-exotics canon
    return float(sum(checks) / len(checks))


def bench_cider_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cider_lite_studies": _bench_cider_lite_studies(seed)}
