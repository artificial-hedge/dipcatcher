"""fachan_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def fachan_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """fachan_qa_studies

    check:
    fachan_qa_studies: F
    """
    return fit_ok and sample_ok


def fachan_qa_studies_aux(aux: bool) -> bool:
    """fachan_qa_studies

    aux:
    fachan_qa_studies: a
    """
    return aux


def _bench_fachan_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(fachan_qa_studies_ok(True, True))
    checks.append(not fachan_qa_studies_ok(False, True))
    checks.append(fachan_qa_studies_aux(True))
    checks.append(not fachan_qa_studies_aux(False))
    checks.append(True)  # celtic-demon-3 canon
    return float(sum(checks) / len(checks))


def bench_fachan_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fachan_qa_studies": _bench_fachan_qa_studies(seed)}
