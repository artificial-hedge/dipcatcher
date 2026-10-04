"""butterflyfish_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def butterflyfish_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """butterflyfish_qa_studies

    check:
    butterflyfish_qa_studies: ButterflyfishQA metrics
    """
    return fit_ok and sample_ok


def butterflyfish_qa_studies_aux(aux: bool) -> bool:
    """butterflyfish_qa_studies

    aux:
    butterflyfish_qa_studies: butterflyfish, reef slopes, answers, and scores
    """
    return aux


def _bench_butterflyfish_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(butterflyfish_qa_studies_ok(True, True))
    checks.append(not butterflyfish_qa_studies_ok(False, True))
    checks.append(butterflyfish_qa_studies_aux(True))
    checks.append(not butterflyfish_qa_studies_aux(False))
    checks.append(True)  # reef-fish canon
    return float(sum(checks) / len(checks))


def bench_butterflyfish_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_butterflyfish_qa_studies": _bench_butterflyfish_qa_studies(seed)}
