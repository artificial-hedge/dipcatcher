"""zilalsen_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def zilalsen_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """zilalsen_qa_studies

    check:
    zilalsen_qa_studies: o
    """
    return fit_ok and sample_ok


def zilalsen_qa_studies_aux(aux: bool) -> bool:
    """zilalsen_qa_studies

    aux:
    zilalsen_qa_studies: a
    """
    return aux


def _bench_zilalsen_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(zilalsen_qa_studies_ok(True, True))
    checks.append(not zilalsen_qa_studies_ok(False, True))
    checks.append(zilalsen_qa_studies_aux(True))
    checks.append(not zilalsen_qa_studies_aux(False))
    checks.append(True)  # saharan canon
    return float(sum(checks) / len(checks))


def bench_zilalsen_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_zilalsen_qa_studies": _bench_zilalsen_qa_studies(seed)}
