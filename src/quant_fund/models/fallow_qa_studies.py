"""fallow_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def fallow_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """fallow_qa_studies

    check:
    fallow_qa_studies: FallowQA metrics
    """
    return fit_ok and sample_ok


def fallow_qa_studies_aux(aux: bool) -> bool:
    """fallow_qa_studies

    aux:
    fallow_qa_studies: fallow deer, parkland herds, answers, and scores
    """
    return aux


def _bench_fallow_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(fallow_qa_studies_ok(True, True))
    checks.append(not fallow_qa_studies_ok(False, True))
    checks.append(fallow_qa_studies_aux(True))
    checks.append(not fallow_qa_studies_aux(False))
    checks.append(True)  # deer canon
    return float(sum(checks) / len(checks))


def bench_fallow_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fallow_qa_studies": _bench_fallow_qa_studies(seed)}
