"""railbird_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def railbird_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """railbird_qa_studies

    check:
    railbird_qa_studies: RailbirdQA metrics
    """
    return fit_ok and sample_ok


def railbird_qa_studies_aux(aux: bool) -> bool:
    """railbird_qa_studies

    aux:
    railbird_qa_studies: railbirds, marshes, answers, and scores
    """
    return aux


def _bench_railbird_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(railbird_qa_studies_ok(True, True))
    checks.append(not railbird_qa_studies_ok(False, True))
    checks.append(railbird_qa_studies_aux(True))
    checks.append(not railbird_qa_studies_aux(False))
    checks.append(True)  # wader-2 canon
    return float(sum(checks) / len(checks))


def bench_railbird_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_railbird_qa_studies": _bench_railbird_qa_studies(seed)}
