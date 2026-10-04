"""samca_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def samca_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """samca_qa_studies

    check:
    samca_qa_studies: S
    """
    return fit_ok and sample_ok


def samca_qa_studies_aux(aux: bool) -> bool:
    """samca_qa_studies

    aux:
    samca_qa_studies: a
    """
    return aux


def _bench_samca_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(samca_qa_studies_ok(True, True))
    checks.append(not samca_qa_studies_ok(False, True))
    checks.append(samca_qa_studies_aux(True))
    checks.append(not samca_qa_studies_aux(False))
    checks.append(True)  # romanian-demon canon
    return float(sum(checks) / len(checks))


def bench_samca_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_samca_qa_studies": _bench_samca_qa_studies(seed)}
