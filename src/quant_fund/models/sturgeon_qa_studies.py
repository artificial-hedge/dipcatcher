"""sturgeon_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def sturgeon_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sturgeon_qa_studies

    check:
    sturgeon_qa_studies: SturgeonQA metrics
    """
    return fit_ok and sample_ok


def sturgeon_qa_studies_aux(aux: bool) -> bool:
    """sturgeon_qa_studies

    aux:
    sturgeon_qa_studies: sturgeons, river bottoms, answers, and scores
    """
    return aux


def _bench_sturgeon_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(sturgeon_qa_studies_ok(True, True))
    checks.append(not sturgeon_qa_studies_ok(False, True))
    checks.append(sturgeon_qa_studies_aux(True))
    checks.append(not sturgeon_qa_studies_aux(False))
    checks.append(True)  # freshwater-fish canon
    return float(sum(checks) / len(checks))


def bench_sturgeon_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sturgeon_qa_studies": _bench_sturgeon_qa_studies(seed)}
