"""tiger_beetle_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def tiger_beetle_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tiger_beetle_qa_studies

    check:
    tiger_beetle_qa_studies: TigerBeetleQA metrics
    """
    return fit_ok and sample_ok


def tiger_beetle_qa_studies_aux(aux: bool) -> bool:
    """tiger_beetle_qa_studies

    aux:
    tiger_beetle_qa_studies: tiger beetles, sandy_paths, answers, and scores
    """
    return aux


def _bench_tiger_beetle_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tiger_beetle_qa_studies_ok(True, True))
    checks.append(not tiger_beetle_qa_studies_ok(False, True))
    checks.append(tiger_beetle_qa_studies_aux(True))
    checks.append(not tiger_beetle_qa_studies_aux(False))
    checks.append(True)  # beetle canon
    return float(sum(checks) / len(checks))


def bench_tiger_beetle_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tiger_beetle_qa_studies": _bench_tiger_beetle_qa_studies(seed)}
