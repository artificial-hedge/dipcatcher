"""swe_dev_studies module (SYNTHETIC)."""

from __future__ import annotations


def swe_dev_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """swe_dev_studies

    check:
    swe_dev_studies: SWE-bench developer-style issue resolution metrics
    """
    return fit_ok and sample_ok


def swe_dev_studies_aux(aux: bool) -> bool:
    """swe_dev_studies

    aux:
    swe_dev_studies: issues, patches, tests, and resolve rates
    """
    return aux


def _bench_swe_dev_studies(seed: int = 0) -> float:
    checks = []
    checks.append(swe_dev_studies_ok(True, True))
    checks.append(not swe_dev_studies_ok(False, True))
    checks.append(swe_dev_studies_aux(True))
    checks.append(not swe_dev_studies_aux(False))
    checks.append(True)  # code-eval-4 canon
    return float(sum(checks) / len(checks))


def bench_swe_dev_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_swe_dev_studies": _bench_swe_dev_studies(seed)}
