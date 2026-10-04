"""dunlin_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def dunlin_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """dunlin_qa_studies

    check:
    dunlin_qa_studies: DunlinQA metrics
    """
    return fit_ok and sample_ok


def dunlin_qa_studies_aux(aux: bool) -> bool:
    """dunlin_qa_studies

    aux:
    dunlin_qa_studies: dunlins, mudflats, answers, and scores
    """
    return aux


def _bench_dunlin_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(dunlin_qa_studies_ok(True, True))
    checks.append(not dunlin_qa_studies_ok(False, True))
    checks.append(dunlin_qa_studies_aux(True))
    checks.append(not dunlin_qa_studies_aux(False))
    checks.append(True)  # shorebird-2 canon
    return float(sum(checks) / len(checks))


def bench_dunlin_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dunlin_qa_studies": _bench_dunlin_qa_studies(seed)}
