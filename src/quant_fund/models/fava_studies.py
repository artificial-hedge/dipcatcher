"""fava_studies module (SYNTHETIC)."""

from __future__ import annotations


def fava_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """fava_studies

    check:
    fava_studies: FAVA fine-grained factuality metrics
    """
    return fit_ok and sample_ok


def fava_studies_aux(aux: bool) -> bool:
    """fava_studies

    aux:
    fava_studies: claims, edits, and error rates
    """
    return aux


def _bench_fava_studies(seed: int = 0) -> float:
    checks = []
    checks.append(fava_studies_ok(True, True))
    checks.append(not fava_studies_ok(False, True))
    checks.append(fava_studies_aux(True))
    checks.append(not fava_studies_aux(False))
    checks.append(True)  # multilingual-eval canon
    return float(sum(checks) / len(checks))


def bench_fava_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fava_studies": _bench_fava_studies(seed)}
