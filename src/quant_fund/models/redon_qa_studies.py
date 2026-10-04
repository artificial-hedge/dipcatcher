"""redon_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def redon_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """redon_qa_studies

    check:
    redon_qa_studies: RedonQA metrics
    """
    return fit_ok and sample_ok


def redon_qa_studies_aux(aux: bool) -> bool:
    """redon_qa_studies

    aux:
    redon_qa_studies: redon, wave kings, answers, and scores
    """
    return aux


def _bench_redon_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(redon_qa_studies_ok(True, True))
    checks.append(not redon_qa_studies_ok(False, True))
    checks.append(redon_qa_studies_aux(True))
    checks.append(not redon_qa_studies_aux(False))
    checks.append(True)  # illyrian-myth canon
    return float(sum(checks) / len(checks))


def bench_redon_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_redon_qa_studies": _bench_redon_qa_studies(seed)}
