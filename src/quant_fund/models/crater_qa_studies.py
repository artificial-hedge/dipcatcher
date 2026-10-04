"""crater_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def crater_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """crater_qa_studies

    check:
    crater_qa_studies: CraterQA metrics
    """
    return fit_ok and sample_ok


def crater_qa_studies_aux(aux: bool) -> bool:
    """crater_qa_studies

    aux:
    crater_qa_studies: craters, impacts, answers, and scores
    """
    return aux


def _bench_crater_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(crater_qa_studies_ok(True, True))
    checks.append(not crater_qa_studies_ok(False, True))
    checks.append(crater_qa_studies_aux(True))
    checks.append(not crater_qa_studies_aux(False))
    checks.append(True)  # landform canon
    return float(sum(checks) / len(checks))


def bench_crater_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_crater_qa_studies": _bench_crater_qa_studies(seed)}
