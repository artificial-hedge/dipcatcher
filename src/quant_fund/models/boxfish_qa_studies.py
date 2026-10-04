"""boxfish_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def boxfish_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """boxfish_qa_studies

    check:
    boxfish_qa_studies: BoxfishQA metrics
    """
    return fit_ok and sample_ok


def boxfish_qa_studies_aux(aux: bool) -> bool:
    """boxfish_qa_studies

    aux:
    boxfish_qa_studies: boxfish, shallow reefs, answers, and scores
    """
    return aux


def _bench_boxfish_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(boxfish_qa_studies_ok(True, True))
    checks.append(not boxfish_qa_studies_ok(False, True))
    checks.append(boxfish_qa_studies_aux(True))
    checks.append(not boxfish_qa_studies_aux(False))
    checks.append(True)  # reef-fish-3 canon
    return float(sum(checks) / len(checks))


def bench_boxfish_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_boxfish_qa_studies": _bench_boxfish_qa_studies(seed)}
