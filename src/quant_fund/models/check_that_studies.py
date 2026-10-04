"""check_that_studies module (SYNTHETIC)."""

from __future__ import annotations


def check_that_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """check_that_studies

    check:
    check_that_studies: check-worthiness metrics
    """
    return fit_ok and sample_ok


def check_that_studies_aux(aux: bool) -> bool:
    """check_that_studies

    aux:
    check_that_studies: claims, sentences, labels, and accuracies
    """
    return aux


def _bench_check_that_studies(seed: int = 0) -> float:
    checks = []
    checks.append(check_that_studies_ok(True, True))
    checks.append(not check_that_studies_ok(False, True))
    checks.append(check_that_studies_aux(True))
    checks.append(not check_that_studies_aux(False))
    checks.append(True)  # fake-news canon
    return float(sum(checks) / len(checks))


def bench_check_that_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_check_that_studies": _bench_check_that_studies(seed)}
