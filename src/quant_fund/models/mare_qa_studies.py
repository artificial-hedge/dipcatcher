"""mare_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def mare_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mare_qa_studies

    check:
    mare_qa_studies: MareQA metrics
    """
    return fit_ok and sample_ok


def mare_qa_studies_aux(aux: bool) -> bool:
    """mare_qa_studies

    aux:
    mare_qa_studies: mares, night riders, answers, and scores
    """
    return aux


def _bench_mare_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mare_qa_studies_ok(True, True))
    checks.append(not mare_qa_studies_ok(False, True))
    checks.append(mare_qa_studies_aux(True))
    checks.append(not mare_qa_studies_aux(False))
    checks.append(True)  # scandinavian-folk canon
    return float(sum(checks) / len(checks))


def bench_mare_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mare_qa_studies": _bench_mare_qa_studies(seed)}
