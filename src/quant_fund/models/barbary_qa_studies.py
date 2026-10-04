"""barbary_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def barbary_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """barbary_qa_studies

    check:
    barbary_qa_studies: BarbaryQA metrics
    """
    return fit_ok and sample_ok


def barbary_qa_studies_aux(aux: bool) -> bool:
    """barbary_qa_studies

    aux:
    barbary_qa_studies: barbary sheep, atlas cliffs, answers, and scores
    """
    return aux


def _bench_barbary_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(barbary_qa_studies_ok(True, True))
    checks.append(not barbary_qa_studies_ok(False, True))
    checks.append(barbary_qa_studies_aux(True))
    checks.append(not barbary_qa_studies_aux(False))
    checks.append(True)  # alpine-ridgeline canon
    return float(sum(checks) / len(checks))


def bench_barbary_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_barbary_qa_studies": _bench_barbary_qa_studies(seed)}
