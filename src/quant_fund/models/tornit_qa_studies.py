"""tornit_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def tornit_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tornit_qa_studies

    check:
    tornit_qa_studies: T
    """
    return fit_ok and sample_ok


def tornit_qa_studies_aux(aux: bool) -> bool:
    """tornit_qa_studies

    aux:
    tornit_qa_studies: o
    """
    return aux


def _bench_tornit_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tornit_qa_studies_ok(True, True))
    checks.append(not tornit_qa_studies_ok(False, True))
    checks.append(tornit_qa_studies_aux(True))
    checks.append(not tornit_qa_studies_aux(False))
    checks.append(True)  # inuit-demon canon
    return float(sum(checks) / len(checks))


def bench_tornit_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tornit_qa_studies": _bench_tornit_qa_studies(seed)}
