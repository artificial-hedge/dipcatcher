"""swe_bench_studies module (SYNTHETIC)."""

from __future__ import annotations


def swe_bench_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """swe_bench_studies

    check:
    swe_bench_studies: SWE-bench issue repos/patches and resolutions
    """
    return fit_ok and sample_ok


def swe_bench_studies_aux(aux: bool) -> bool:
    """swe_bench_studies

    aux:
    swe_bench_studies: fail-to-pass test suites/diffs and outcomes
    """
    return aux


def _bench_swe_bench_studies(seed: int = 0) -> float:
    checks = []
    checks.append(swe_bench_studies_ok(True, True))
    checks.append(not swe_bench_studies_ok(False, True))
    checks.append(swe_bench_studies_aux(True))
    checks.append(not swe_bench_studies_aux(False))
    checks.append(True)  # agentic-eval canon
    return float(sum(checks) / len(checks))


def bench_swe_bench_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_swe_bench_studies": _bench_swe_bench_studies(seed)}
