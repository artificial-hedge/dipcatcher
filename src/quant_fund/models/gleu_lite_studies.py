"""gleu_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def gleu_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """gleu_lite_studies

    check:
    gleu_lite_studies: GLEU grammar metrics
    """
    return fit_ok and sample_ok


def gleu_lite_studies_aux(aux: bool) -> bool:
    """gleu_lite_studies

    aux:
    gleu_lite_studies: sources, references, candidates, and scores
    """
    return aux


def _bench_gleu_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(gleu_lite_studies_ok(True, True))
    checks.append(not gleu_lite_studies_ok(False, True))
    checks.append(gleu_lite_studies_aux(True))
    checks.append(not gleu_lite_studies_aux(False))
    checks.append(True)  # metric-exotics canon
    return float(sum(checks) / len(checks))


def bench_gleu_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gleu_lite_studies": _bench_gleu_lite_studies(seed)}
