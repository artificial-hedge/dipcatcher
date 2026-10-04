"""olympic_bench_studies module (SYNTHETIC)."""

from __future__ import annotations


def olympic_bench_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """olympic_bench_studies

    check:
    olympic_bench_studies: OlympiadBench metrics
    """
    return fit_ok and sample_ok


def olympic_bench_studies_aux(aux: bool) -> bool:
    """olympic_bench_studies

    aux:
    olympic_bench_studies: problems, solutions, answers, and scores
    """
    return aux


def _bench_olympic_bench_studies(seed: int = 0) -> float:
    checks = []
    checks.append(olympic_bench_studies_ok(True, True))
    checks.append(not olympic_bench_studies_ok(False, True))
    checks.append(olympic_bench_studies_aux(True))
    checks.append(not olympic_bench_studies_aux(False))
    checks.append(True)  # challenge-benchmark canon
    return float(sum(checks) / len(checks))


def bench_olympic_bench_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_olympic_bench_studies": _bench_olympic_bench_studies(seed)}
