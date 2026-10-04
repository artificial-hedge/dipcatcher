"""tarantula_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def tarantula_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tarantula_qa_studies

    check:
    tarantula_qa_studies: TarantulaQA metrics
    """
    return fit_ok and sample_ok


def tarantula_qa_studies_aux(aux: bool) -> bool:
    """tarantula_qa_studies

    aux:
    tarantula_qa_studies: tarantulas, desert burrows, answers, and scores
    """
    return aux


def _bench_tarantula_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tarantula_qa_studies_ok(True, True))
    checks.append(not tarantula_qa_studies_ok(False, True))
    checks.append(tarantula_qa_studies_aux(True))
    checks.append(not tarantula_qa_studies_aux(False))
    checks.append(True)  # spider canon
    return float(sum(checks) / len(checks))


def bench_tarantula_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tarantula_qa_studies": _bench_tarantula_qa_studies(seed)}
