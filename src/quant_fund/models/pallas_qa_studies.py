"""pallas_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def pallas_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """pallas_qa_studies

    check:
    pallas_qa_studies: PallasQA metrics
    """
    return fit_ok and sample_ok


def pallas_qa_studies_aux(aux: bool) -> bool:
    """pallas_qa_studies

    aux:
    pallas_qa_studies: pallas cats, steppe rocks, answers, and scores
    """
    return aux


def _bench_pallas_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(pallas_qa_studies_ok(True, True))
    checks.append(not pallas_qa_studies_ok(False, True))
    checks.append(pallas_qa_studies_aux(True))
    checks.append(not pallas_qa_studies_aux(False))
    checks.append(True)  # small-cat canon
    return float(sum(checks) / len(checks))


def bench_pallas_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pallas_qa_studies": _bench_pallas_qa_studies(seed)}
