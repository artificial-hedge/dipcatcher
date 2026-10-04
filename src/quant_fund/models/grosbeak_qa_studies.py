"""grosbeak_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def grosbeak_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """grosbeak_qa_studies

    check:
    grosbeak_qa_studies: GrosbeakQA metrics
    """
    return fit_ok and sample_ok


def grosbeak_qa_studies_aux(aux: bool) -> bool:
    """grosbeak_qa_studies

    aux:
    grosbeak_qa_studies: grosbeaks, thickets, answers, and scores
    """
    return aux


def _bench_grosbeak_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(grosbeak_qa_studies_ok(True, True))
    checks.append(not grosbeak_qa_studies_ok(False, True))
    checks.append(grosbeak_qa_studies_aux(True))
    checks.append(not grosbeak_qa_studies_aux(False))
    checks.append(True)  # songbird-2 canon
    return float(sum(checks) / len(checks))


def bench_grosbeak_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_grosbeak_qa_studies": _bench_grosbeak_qa_studies(seed)}
