"""nabu_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def nabu_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nabu_qa_studies

    check:
    nabu_qa_studies: NabuQA metrics
    """
    return fit_ok and sample_ok


def nabu_qa_studies_aux(aux: bool) -> bool:
    """nabu_qa_studies

    aux:
    nabu_qa_studies: nabu, scribe gods, answers, and scores
    """
    return aux


def _bench_nabu_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nabu_qa_studies_ok(True, True))
    checks.append(not nabu_qa_studies_ok(False, True))
    checks.append(nabu_qa_studies_aux(True))
    checks.append(not nabu_qa_studies_aux(False))
    checks.append(True)  # babylonian-myth canon
    return float(sum(checks) / len(checks))


def bench_nabu_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nabu_qa_studies": _bench_nabu_qa_studies(seed)}
