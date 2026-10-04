"""crag_bench_studies module (SYNTHETIC)."""

from __future__ import annotations


def crag_bench_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """crag_bench_studies

    check:
    crag_bench_studies: CRAG metrics
    """
    return fit_ok and sample_ok


def crag_bench_studies_aux(aux: bool) -> bool:
    """crag_bench_studies

    aux:
    crag_bench_studies: questions, retrievals, answers, and scores
    """
    return aux


def _bench_crag_bench_studies(seed: int = 0) -> float:
    checks = []
    checks.append(crag_bench_studies_ok(True, True))
    checks.append(not crag_bench_studies_ok(False, True))
    checks.append(crag_bench_studies_aux(True))
    checks.append(not crag_bench_studies_aux(False))
    checks.append(True)  # RAG-eval canon
    return float(sum(checks) / len(checks))


def bench_crag_bench_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_crag_bench_studies": _bench_crag_bench_studies(seed)}
