"""sheartail_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def sheartail_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sheartail_qa_studies

    check:
    sheartail_qa_studies: SheartailQA metrics
    """
    return fit_ok and sample_ok


def sheartail_qa_studies_aux(aux: bool) -> bool:
    """sheartail_qa_studies

    aux:
    sheartail_qa_studies: sheartails, valleys, answers, and scores
    """
    return aux


def _bench_sheartail_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(sheartail_qa_studies_ok(True, True))
    checks.append(not sheartail_qa_studies_ok(False, True))
    checks.append(sheartail_qa_studies_aux(True))
    checks.append(not sheartail_qa_studies_aux(False))
    checks.append(True)  # hummingbird-2 canon
    return float(sum(checks) / len(checks))


def bench_sheartail_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sheartail_qa_studies": _bench_sheartail_qa_studies(seed)}
