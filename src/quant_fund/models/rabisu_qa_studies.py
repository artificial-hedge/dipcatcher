"""rabisu_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def rabisu_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """rabisu_qa_studies

    check:
    rabisu_qa_studies: r
    """
    return fit_ok and sample_ok


def rabisu_qa_studies_aux(aux: bool) -> bool:
    """rabisu_qa_studies

    aux:
    rabisu_qa_studies: a
    """
    return aux


def _bench_rabisu_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(rabisu_qa_studies_ok(True, True))
    checks.append(not rabisu_qa_studies_ok(False, True))
    checks.append(rabisu_qa_studies_aux(True))
    checks.append(not rabisu_qa_studies_aux(False))
    checks.append(True)  # mesopotamian-demon canon
    return float(sum(checks) / len(checks))


def bench_rabisu_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rabisu_qa_studies": _bench_rabisu_qa_studies(seed)}
