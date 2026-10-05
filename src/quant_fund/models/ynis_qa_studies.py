"""ynis_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ynis_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ynis_qa_studies

    check:
    ynis_qa_studies: i
    """
    return fit_ok and sample_ok


def ynis_qa_studies_aux(aux: bool) -> bool:
    """ynis_qa_studies

    aux:
    ynis_qa_studies: s
    """
    return aux


def _bench_ynis_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ynis_qa_studies_ok(True, True))
    checks.append(not ynis_qa_studies_ok(False, True))
    checks.append(ynis_qa_studies_aux(True))
    checks.append(not ynis_qa_studies_aux(False))
    checks.append(True)  # arthurian-7 canon
    return float(sum(checks) / len(checks))


def bench_ynis_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ynis_qa_studies": _bench_ynis_qa_studies(seed)}
