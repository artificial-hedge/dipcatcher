"""fritillary_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def fritillary_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """fritillary_qa_studies

    check:
    fritillary_qa_studies: FritillaryQA metrics
    """
    return fit_ok and sample_ok


def fritillary_qa_studies_aux(aux: bool) -> bool:
    """fritillary_qa_studies

    aux:
    fritillary_qa_studies: fritillaries, violets, answers, and scores
    """
    return aux


def _bench_fritillary_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(fritillary_qa_studies_ok(True, True))
    checks.append(not fritillary_qa_studies_ok(False, True))
    checks.append(fritillary_qa_studies_aux(True))
    checks.append(not fritillary_qa_studies_aux(False))
    checks.append(True)  # butterfly canon
    return float(sum(checks) / len(checks))


def bench_fritillary_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fritillary_qa_studies": _bench_fritillary_qa_studies(seed)}
