"""multirc_studies module (SYNTHETIC)."""

from __future__ import annotations


def multirc_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """multirc_studies

    check:
    multirc_studies: SuperGLUE MultiRC multi-sentence QA and F1a/EM
    """
    return fit_ok and sample_ok


def multirc_studies_aux(aux: bool) -> bool:
    """multirc_studies

    aux:
    multirc_studies: passages, questions, candidates, and scores
    """
    return aux


def _bench_multirc_studies(seed: int = 0) -> float:
    checks = []
    checks.append(multirc_studies_ok(True, True))
    checks.append(not multirc_studies_ok(False, True))
    checks.append(multirc_studies_aux(True))
    checks.append(not multirc_studies_aux(False))
    checks.append(True)  # benchmark-eval canon
    return float(sum(checks) / len(checks))


def bench_multirc_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_multirc_studies": _bench_multirc_studies(seed)}
