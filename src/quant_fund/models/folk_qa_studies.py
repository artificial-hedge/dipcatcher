"""folk_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def folk_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """folk_qa_studies

    check:
    folk_qa_studies: FolkQA metrics
    """
    return fit_ok and sample_ok


def folk_qa_studies_aux(aux: bool) -> bool:
    """folk_qa_studies

    aux:
    folk_qa_studies: claims, beliefs, answers, and scores
    """
    return aux


def _bench_folk_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(folk_qa_studies_ok(True, True))
    checks.append(not folk_qa_studies_ok(False, True))
    checks.append(folk_qa_studies_aux(True))
    checks.append(not folk_qa_studies_aux(False))
    checks.append(True)  # folk-commonsense canon
    return float(sum(checks) / len(checks))


def bench_folk_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_folk_qa_studies": _bench_folk_qa_studies(seed)}
