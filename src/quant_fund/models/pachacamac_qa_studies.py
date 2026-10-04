"""pachacamac_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def pachacamac_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """pachacamac_qa_studies

    check:
    pachacamac_qa_studies: PachacamacQA metrics
    """
    return fit_ok and sample_ok


def pachacamac_qa_studies_aux(aux: bool) -> bool:
    """pachacamac_qa_studies

    aux:
    pachacamac_qa_studies: pachacamac, earth shapers, answers, and scores
    """
    return aux


def _bench_pachacamac_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(pachacamac_qa_studies_ok(True, True))
    checks.append(not pachacamac_qa_studies_ok(False, True))
    checks.append(pachacamac_qa_studies_aux(True))
    checks.append(not pachacamac_qa_studies_aux(False))
    checks.append(True)  # incan-myth-2 canon
    return float(sum(checks) / len(checks))


def bench_pachacamac_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pachacamac_qa_studies": _bench_pachacamac_qa_studies(seed)}
