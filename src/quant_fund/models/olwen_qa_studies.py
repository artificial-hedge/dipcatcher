"""olwen_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def olwen_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """olwen_qa_studies

    check:
    olwen_qa_studies: g
    """
    return fit_ok and sample_ok


def olwen_qa_studies_aux(aux: bool) -> bool:
    """olwen_qa_studies

    aux:
    olwen_qa_studies: i
    """
    return aux


def _bench_olwen_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(olwen_qa_studies_ok(True, True))
    checks.append(not olwen_qa_studies_ok(False, True))
    checks.append(olwen_qa_studies_aux(True))
    checks.append(not olwen_qa_studies_aux(False))
    checks.append(True)  # arthurian-6 canon
    return float(sum(checks) / len(checks))


def bench_olwen_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_olwen_qa_studies": _bench_olwen_qa_studies(seed)}
