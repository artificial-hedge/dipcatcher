"""xstest_studies module (SYNTHETIC)."""

from __future__ import annotations


def xstest_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """xstest_studies

    check:
    xstest_studies: XSTest safe/unsafe prompt sets and refusal
    """
    return fit_ok and sample_ok


def xstest_studies_aux(aux: bool) -> bool:
    """xstest_studies

    aux:
    xstest_studies: over-refusal contrast items/scores and flags
    """
    return aux


def _bench_xstest_studies(seed: int = 0) -> float:
    checks = []
    checks.append(xstest_studies_ok(True, True))
    checks.append(not xstest_studies_ok(False, True))
    checks.append(xstest_studies_aux(True))
    checks.append(not xstest_studies_aux(False))
    checks.append(True)  # safety-eval canon
    return float(sum(checks) / len(checks))


def bench_xstest_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_xstest_studies": _bench_xstest_studies(seed)}
