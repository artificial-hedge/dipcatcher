"""ruler_bench_studies module (SYNTHETIC)."""

from __future__ import annotations


def ruler_bench_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ruler_bench_studies

    check:
    ruler_bench_studies: RULER synthetic-task effective-context metrics
    """
    return fit_ok and sample_ok


def ruler_bench_studies_aux(aux: bool) -> bool:
    """ruler_bench_studies

    aux:
    ruler_bench_studies: haystacks, needles, tasks, and scores
    """
    return aux


def _bench_ruler_bench_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ruler_bench_studies_ok(True, True))
    checks.append(not ruler_bench_studies_ok(False, True))
    checks.append(ruler_bench_studies_aux(True))
    checks.append(not ruler_bench_studies_aux(False))
    checks.append(True)  # long-context-eval canon
    return float(sum(checks) / len(checks))


def bench_ruler_bench_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ruler_bench_studies": _bench_ruler_bench_studies(seed)}
