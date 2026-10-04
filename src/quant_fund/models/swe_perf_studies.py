"""swe_perf_studies module (SYNTHETIC)."""

from __future__ import annotations


def swe_perf_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """swe_perf_studies

    check:
    swe_perf_studies: SWE-Perf performance-targeted code change eval
    """
    return fit_ok and sample_ok


def swe_perf_studies_aux(aux: bool) -> bool:
    """swe_perf_studies

    aux:
    swe_perf_studies: repos, diffs, and speedup scores
    """
    return aux


def _bench_swe_perf_studies(seed: int = 0) -> float:
    checks = []
    checks.append(swe_perf_studies_ok(True, True))
    checks.append(not swe_perf_studies_ok(False, True))
    checks.append(swe_perf_studies_aux(True))
    checks.append(not swe_perf_studies_aux(False))
    checks.append(True)  # code-eval canon
    return float(sum(checks) / len(checks))


def bench_swe_perf_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_swe_perf_studies": _bench_swe_perf_studies(seed)}
