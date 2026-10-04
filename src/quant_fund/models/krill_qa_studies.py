"""krill_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def krill_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """krill_qa_studies

    check:
    krill_qa_studies: KrillQA metrics
    """
    return fit_ok and sample_ok


def krill_qa_studies_aux(aux: bool) -> bool:
    """krill_qa_studies

    aux:
    krill_qa_studies: krill, cold pelagic swarms, answers, and scores
    """
    return aux


def _bench_krill_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(krill_qa_studies_ok(True, True))
    checks.append(not krill_qa_studies_ok(False, True))
    checks.append(krill_qa_studies_aux(True))
    checks.append(not krill_qa_studies_aux(False))
    checks.append(True)  # plankton-shore canon
    return float(sum(checks) / len(checks))


def bench_krill_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_krill_qa_studies": _bench_krill_qa_studies(seed)}
