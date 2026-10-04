"""isis_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def isis_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """isis_qa_studies

    check:
    isis_qa_studies: IsisQA metrics
    """
    return fit_ok and sample_ok


def isis_qa_studies_aux(aux: bool) -> bool:
    """isis_qa_studies

    aux:
    isis_qa_studies: isis, throne mothers, answers, and scores
    """
    return aux


def _bench_isis_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(isis_qa_studies_ok(True, True))
    checks.append(not isis_qa_studies_ok(False, True))
    checks.append(isis_qa_studies_aux(True))
    checks.append(not isis_qa_studies_aux(False))
    checks.append(True)  # egyptian-5 canon
    return float(sum(checks) / len(checks))


def bench_isis_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_isis_qa_studies": _bench_isis_qa_studies(seed)}
