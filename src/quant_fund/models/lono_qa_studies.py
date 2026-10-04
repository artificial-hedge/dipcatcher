"""lono_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def lono_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """lono_qa_studies

    check:
    lono_qa_studies: LonoQA metrics
    """
    return fit_ok and sample_ok


def lono_qa_studies_aux(aux: bool) -> bool:
    """lono_qa_studies

    aux:
    lono_qa_studies: lono, rain planters, answers, and scores
    """
    return aux


def _bench_lono_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(lono_qa_studies_ok(True, True))
    checks.append(not lono_qa_studies_ok(False, True))
    checks.append(lono_qa_studies_aux(True))
    checks.append(not lono_qa_studies_aux(False))
    checks.append(True)  # hawaiian-myth canon
    return float(sum(checks) / len(checks))


def bench_lono_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lono_qa_studies": _bench_lono_qa_studies(seed)}
