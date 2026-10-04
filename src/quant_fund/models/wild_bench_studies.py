"""wild_bench_studies module (SYNTHETIC)."""

from __future__ import annotations


def wild_bench_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """wild_bench_studies

    check:
    wild_bench_studies: WildBench task-completion quality and win rates
    """
    return fit_ok and sample_ok


def wild_bench_studies_aux(aux: bool) -> bool:
    """wild_bench_studies

    aux:
    wild_bench_studies: wild queries, rubric scores, and comparisons
    """
    return aux


def _bench_wild_bench_studies(seed: int = 0) -> float:
    checks = []
    checks.append(wild_bench_studies_ok(True, True))
    checks.append(not wild_bench_studies_ok(False, True))
    checks.append(wild_bench_studies_aux(True))
    checks.append(not wild_bench_studies_aux(False))
    checks.append(True)  # benchmark-eval canon
    return float(sum(checks) / len(checks))


def bench_wild_bench_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_wild_bench_studies": _bench_wild_bench_studies(seed)}
