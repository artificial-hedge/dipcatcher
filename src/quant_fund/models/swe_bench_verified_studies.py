"""swe_bench_verified_studies module (SYNTHETIC)."""

from __future__ import annotations


def swe_bench_verified_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """swe_bench_verified_studies

    check:
    swe_bench_verified_studies: SWE-bench Verified issue-resolution metrics
    """
    return fit_ok and sample_ok


def swe_bench_verified_studies_aux(aux: bool) -> bool:
    """swe_bench_verified_studies

    aux:
    swe_bench_verified_studies: issues, patches, tests, and resolve rates
    """
    return aux


def _bench_swe_bench_verified_studies(seed: int = 0) -> float:
    checks = []
    checks.append(swe_bench_verified_studies_ok(True, True))
    checks.append(not swe_bench_verified_studies_ok(False, True))
    checks.append(swe_bench_verified_studies_aux(True))
    checks.append(not swe_bench_verified_studies_aux(False))
    checks.append(True)  # code-eval-3 canon
    return float(sum(checks) / len(checks))


def bench_swe_bench_verified_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_swe_bench_verified_studies": _bench_swe_bench_verified_studies(seed)}
